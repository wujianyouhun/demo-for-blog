"""五种 U-Net 变体；均输出二分类分割 logits，不在模型内部做 sigmoid。"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class Conv(nn.Module):
    def __init__(self, a: int, b: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(True),
                                 nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(True))
    def forward(self, x): return self.net(x)


class ResidualConv(nn.Module):
    def __init__(self, a: int, b: int):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(True),
                                  nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b))
        self.skip = nn.Identity() if a == b else nn.Sequential(nn.Conv2d(a, b, 1, bias=False), nn.BatchNorm2d(b))
        self.act = nn.ReLU(True)
    def forward(self, x): return self.act(self.main(x) + self.skip(x))


class UNetCore(nn.Module):
    """普通 U-Net 和 ResUNet 共用的编码器/解码器。"""
    def __init__(self, block=Conv, in_channels=3, base=32):
        super().__init__(); w = [base, base*2, base*4, base*8]
        self.e1, self.e2, self.e3, self.e4 = block(in_channels,w[0]), block(w[0],w[1]), block(w[1],w[2]), block(w[2],w[3])
        self.pool = nn.MaxPool2d(2); self.mid = block(w[3], w[3]*2)
        self.d4, self.d3, self.d2, self.d1 = block(w[3]*3,w[3]), block(w[2]*3,w[2]), block(w[1]*3,w[1]), block(w[0]*3,w[0])
        self.head = nn.Conv2d(w[0], 1, 1)
    def skips(self, x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3)); return e1,e2,e3,e4,self.mid(self.pool(e4))
    def decode(self, e1,e2,e3,e4,x):
        x=self.d4(torch.cat((e4,_up(x,e4)),1)); x=self.d3(torch.cat((e3,_up(x,e3)),1)); x=self.d2(torch.cat((e2,_up(x,e2)),1)); return self.head(self.d1(torch.cat((e1,_up(x,e1)),1)))
    def forward(self,x): return self.decode(*self.skips(x))


class ResUNet(UNetCore):
    def __init__(self, **kw): super().__init__(block=ResidualConv, **kw)


class Gate(nn.Module):
    def __init__(self, skip_ch, gate_ch):
        super().__init__(); m=max(1,skip_ch//2); self.s=nn.Conv2d(skip_ch,m,1); self.g=nn.Conv2d(gate_ch,m,1); self.a=nn.Sequential(nn.ReLU(True),nn.Conv2d(m,1,1),nn.Sigmoid())
    def forward(self, skip, gate): return skip * self.a(self.s(skip)+_up(self.g(gate),skip))


class AttentionUNet(UNetCore):
    def __init__(self, **kw):
        super().__init__(**kw); b=kw.get("base",32); self.g4,self.g3,self.g2,self.g1=Gate(b*8,b*16),Gate(b*4,b*8),Gate(b*2,b*4),Gate(b,b*2)
    def forward(self,x):
        e1,e2,e3,e4,x=self.skips(x)
        x=self.d4(torch.cat((self.g4(e4,x),_up(x,e4)),1)); x=self.d3(torch.cat((self.g3(e3,x),_up(x,e3)),1)); x=self.d2(torch.cat((self.g2(e2,x),_up(x,e2)),1)); return self.head(self.d1(torch.cat((self.g1(e1,x),_up(x,e1)),1)))


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, base=32):
        super().__init__(); b=base; self.pool=nn.MaxPool2d(2)
        self.x00,self.x10,self.x20,self.x30=Conv(in_channels,b),Conv(b,b*2),Conv(b*2,b*4),Conv(b*4,b*8)
        self.x01,self.x11,self.x21=Conv(b*3,b),Conv(b*6,b*2),Conv(b*12,b*4)
        self.x02,self.x12=Conv(b*4,b),Conv(b*8,b*2); self.x03=Conv(b*5,b); self.head=nn.Conv2d(b,1,1)
    def forward(self,x):
        a=self.x00(x); b=self.x10(self.pool(a)); c=self.x20(self.pool(b)); d=self.x30(self.pool(c))
        a1=self.x01(torch.cat((a,_up(b,a)),1)); b1=self.x11(torch.cat((b,_up(c,b)),1)); c1=self.x21(torch.cat((c,_up(d,c)),1))
        a2=self.x02(torch.cat((a,a1,_up(b1,a)),1)); b2=self.x12(torch.cat((b,b1,_up(c1,b)),1)); a3=self.x03(torch.cat((a,a1,a2,_up(b2,a)),1)); return self.head(a3)


class TransformerBlock(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__(); self.norm1=nn.LayerNorm(channels); self.attn=nn.MultiheadAttention(channels,heads,batch_first=True); self.norm2=nn.LayerNorm(channels); self.ff=nn.Sequential(nn.Linear(channels,channels*2),nn.GELU(),nn.Linear(channels*2,channels))
    def forward(self,x):
        n,c,h,w=x.shape; z=x.flatten(2).transpose(1,2); q=self.norm1(z); z=z+self.attn(q,q,q,need_weights=False)[0]; z=z+self.ff(self.norm2(z)); return z.transpose(1,2).reshape(n,c,h,w)


class TransUNet(UNetCore):
    def __init__(self, **kw):
        super().__init__(**kw); self.transformer=TransformerBlock(kw.get("base",32)*16)
    def forward(self,x):
        e1,e2,e3,e4,z=self.skips(x); return self.decode(e1,e2,e3,e4,self.transformer(z))


class WindowBlock(nn.Module):
    """Swin 的窗口自注意力简化实现：避免整幅高分影像的平方级显存。"""
    def __init__(self, channels, window=8):
        super().__init__(); self.window=window; self.block=TransformerBlock(channels)
    def forward(self,x):
        n,c,h,w=x.shape; p1=(self.window-h%self.window)%self.window; p2=(self.window-w%self.window)%self.window
        z=F.pad(x,(0,p2,0,p1)); hh,ww=z.shape[-2:]; z=z.reshape(n,c,hh//self.window,self.window,ww//self.window,self.window).permute(0,2,4,1,3,5).reshape(-1,c,self.window,self.window)
        z=self.block(z).reshape(n,hh//self.window,ww//self.window,c,self.window,self.window).permute(0,3,1,4,2,5).reshape(n,c,hh,ww); return z[:,:,:h,:w]


class SwinUNet(UNetCore):
    def __init__(self, **kw):
        super().__init__(**kw); self.swin=WindowBlock(kw.get("base",32)*16)
    def forward(self,x):
        e1,e2,e3,e4,z=self.skips(x); return self.decode(e1,e2,e3,e4,self.swin(z))


MODELS={"unet":UNetCore,"resunet":ResUNet,"attention_unet":AttentionUNet,"unetpp":UNetPlusPlus,"transunet":TransUNet,"swin_unet":SwinUNet}
def build_model(name: str, in_channels=3, base=32):
    return MODELS[name](in_channels=in_channels, base=base)
