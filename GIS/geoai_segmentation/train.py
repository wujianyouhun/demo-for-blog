import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from models.unet_plus_plus import UNetPlusPlus
from models.deeplabv3_plus import DeepLabV3Plus

class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        self.mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = torch.tensor(Image.open(self.img_paths[idx]).resize((512,512))).permute(2,0,1)/255.0
        mask = torch.tensor(Image.open(self.mask_paths[idx]).resize((512,512))).unsqueeze(0)/255.0
        return img, mask

# 配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNetPlusPlus().to(device)  # 换成 DeepLabV3Plus() 即可
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
dataset = CustomDataset(img_dir='./data/images', mask_dir='./data/masks')
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 训练
model.train()
for epoch in range(50):
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'best_model.pth')