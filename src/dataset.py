import os
from pathlib import Path
import random
import hydra
import pandas as pd
import torch
from torchvision import transforms


from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class ChestXrayDataset(Dataset):

    def __init__(self, data_list, data_dir, label_map ,augment=True, augment_config = False, crop_size =256):

        super().__init__()

        self.data_dir = data_dir
        self.label_map = label_map

        df = pd.read_csv(data_list)
        
        self.paths = df['image_path']
        self.labels = list(map(str, df['label']))

        self.input_size = crop_size


        self.transforms = get_image_transforms(self.input_size, augment, augment_config)

    def __getitem__(self, index):

        img_path = os.path.join(self.data_dir, self.paths[index])

        image = Image.open(rf"{img_path}").convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)

        label = torch.tensor(int(self.label_map[self.labels[index]]))

        return image, label

    def __len__(self):
        return len(self.labels)


class ChestXrayDatamodule(LightningDataModule):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def setup(self, stage=None) -> None:

        self.train_dataset = ChestXrayDataset(
            self.config.train_list,
            self.config.data_dir,
            self.config.label_map,
            augment=True,
            crop_size=self.config.crop_size,
            augment_config=self.config.augmentation,
        )

        self.val_dataset = ChestXrayDataset(
            self.config.val_list,
            self.config.data_dir,
            self.config.label_map,
            augment=False,
            crop_size=self.config.crop_size,
            augment_config=None,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
        )



def get_image_transforms(input_size, augment, augment_config=None):
    transforms_lists= []
    if augment:
        transforms_lists += [transforms.Resize([input_size, input_size])]
        for aug in augment_config:
            transforms_lists += [hydra.utils.instantiate(augment_config[aug])]
        transforms_lists += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    else:
        transforms_lists += [
            transforms.Resize([input_size, input_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    transforms_lists = transforms.Compose(transforms_lists)
    return transforms_lists




# import hydra
# from omegaconf import DictConfig, OmegaConf
# @hydra.main(config_path="configs", config_name="default")
# def main(config: DictConfig) -> None:
#     print(config.dataset)
#     datamodule = SportDatamodule(config.dataset)
#     datamodule.setup()
#     train_loader = datamodule.train_dataloader()
#     batch = next(iter(train_loader))
#     print(batch[0])
#     print(batch[1])

# if __name__ == "__main__":
#     main()