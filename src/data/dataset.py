from omegaconf import DictConfig

from src.data.coco import CocoDataModule
from src.data.img_dataset import DataModule

def get_data_module(data_cfg: DictConfig):
    
    if data_cfg.name == "coco":
        print(f"Loading data module for {data_cfg.name} dataset")
        return CocoDataModule(base_dir=data_cfg.data_dir,
                              img_size=data_cfg.img_size,
                              batch_size=data_cfg.batch_size,
                              num_workers=data_cfg.num_workers)
    elif data_cfg.name == "img":
        print(f"Loading data module for {data_cfg.name} dataset")
        return DataModule(data_dir=data_cfg.data_dir,
                          batch_size=data_cfg.batch_size,
                          img_size=data_cfg.img_size,
                          train_ratio=data_cfg.train_ratio,
                          num_workers=data_cfg.num_workers)
    else:
        return DataModule(data_cfg)