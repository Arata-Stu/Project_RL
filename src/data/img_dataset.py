import os
import glob
import random
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def split_dataset(data_list, train_ratio=0.8):
    random.shuffle(data_list)
    split_idx = int(len(data_list) * train_ratio)
    return data_list[:split_idx], data_list[split_idx:]

class ImgDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, img_size=64, train_ratio=0.8, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        image_paths = glob.glob(os.path.join(self.data_dir, "ep*/step*.png"))
        print(f"Found {len(image_paths)} images in {self.data_dir}")
        train_paths, val_paths = split_dataset(image_paths, self.train_ratio)
        
        self.train_dataset = ImgDataset(train_paths, transform=self.transform)
        self.val_dataset = ImgDataset(val_paths, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

# 例: DataModule の使用
if __name__ == "__main__":
    data_module = DataModule("./output_dir", batch_size=32)
    data_module.setup()
    print(f"Train dataset size: {len(data_module.train_dataset)}")
    print(f"Validation dataset size: {len(data_module.val_dataset)}")
