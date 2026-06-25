"""
EuroSAT 数据集下载 & 整理脚本
EuroSAT: 27000 张 64x64 Sentinel-2 卫星图像，共 10 个地物类别
用法:  conda activate geoai && python scripts/download_dataset.py
"""
import os, zipfile, shutil, urllib.request, random
from pathlib import Path
from tqdm import tqdm

DATA_ROOT   = Path(__file__).resolve().parent.parent / "data"
RAW_DIR     = DATA_ROOT / "raw"
DATASET_DIR = DATA_ROOT / "processed" / "EuroSAT"
EUROSAT_URL = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
BACKUP_URL  = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
CLASS_NAMES = ["AnnualCrop","Forest","HerbaceousVegetation","Highway",
               "Industrial","Pasture","PermanentCrop","Residential","River","SeaLake"]
SPLIT_RATIOS = {"train":0.7,"val":0.15,"test":0.15}
RANDOM_SEED  = 42

class _Progress(tqdm):
    def update_to(self,b=1,bsize=1,tsize=None):
        if tsize is not None: self.total=tsize
        self.update(b*bsize-self.n)

def download_file(url,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    try:
        with _Progress(unit="B",unit_scale=True,miniters=1,desc=dest.name) as t:
            urllib.request.urlretrieve(url,dest,reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"  ✗ 下载失败: {e}"); return False

def extract_zip(zip_path,dest):
    print(f"📦 解压 {zip_path.name} ...")
    with zipfile.ZipFile(zip_path,"r") as zf: zf.extractall(dest)
    print("  ✓ 解压完成")

def split_dataset(src_root):
    random.seed(RANDOM_SEED)
    print("\n📂 拆分数据集 (train / val / test) ...")
    for split in SPLIT_RATIOS: (DATASET_DIR/split).mkdir(parents=True,exist_ok=True)
    for cls in CLASS_NAMES:
        cls_dir = src_root/cls
        if not cls_dir.exists(): print(f"  ⚠ 未找到: {cls_dir}"); continue
        images = sorted(list(cls_dir.glob("*.jpg"))+list(cls_dir.glob("*.png"))+list(cls_dir.glob("*.tif")))
        random.shuffle(images)
        n=len(images); n_train=int(n*0.7); n_val=int(n*0.15)
        buckets={"train":images[:n_train],"val":images[n_train:n_train+n_val],"test":images[n_train+n_val:]}
        for split,imgs in buckets.items():
            dest_cls=DATASET_DIR/split/cls; dest_cls.mkdir(parents=True,exist_ok=True)
            for img in imgs: shutil.copy2(img,dest_cls/img.name)
        print(f"  {cls:<28} train={len(buckets['train'])} val={len(buckets['val'])} test={len(buckets['test'])}")
    print(f"\n✅ 数据集拆分完成 -> {DATASET_DIR}")

def main():
    RAW_DIR.mkdir(parents=True,exist_ok=True)
    zip_path = RAW_DIR/"EuroSAT.zip"
    if not zip_path.exists():
        print("🌐 下载 EuroSAT ...")
        ok = download_file(EUROSAT_URL,zip_path) or download_file(BACKUP_URL,zip_path)
        if not ok: print("\n❌ 请手动下载并解压到:",RAW_DIR); return
    else: print(f"✓ 已存在: {zip_path}")
    extract_dir = RAW_DIR/"EuroSAT_raw"
    if not extract_dir.exists(): extract_zip(zip_path,extract_dir)
    src_root = None
    for cand in [extract_dir/"EuroSAT",extract_dir/"EuroSAT_RGB",extract_dir]:
        if cand.exists() and any((cand/c).exists() for c in CLASS_NAMES): src_root=cand; break
    if src_root is None:
        for p in extract_dir.rglob("AnnualCrop"): src_root=p.parent; break
    if src_root is None: print("❌ 无法定位数据集根目录"); return
    if not (DATASET_DIR/"train").exists(): split_dataset(src_root)
    else: print("✓ 数据集已拆分，跳过")
    print("\n📊 数据集统计:")
    for split in ["train","val","test"]:
        n=sum(1 for f in (DATASET_DIR/split).rglob("*") if f.is_file())
        print(f"  {split:5s}: {n} 张")

if __name__=="__main__": main()
