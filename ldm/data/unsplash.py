import json
from pathlib import Path
from typing import Callable, Tuple

from torch.utils.data import DataLoader, Dataset, random_split

import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image
from ldm.util import instantiate_from_config
from einops import rearrange


class UnsplashDataset(Dataset):
    def __init__(self, root_dir: str, caption_file: str, image_transforms=()):
        self.root_dir = Path(root_dir)

        with open( self.root_dir / Path(caption_file), mode="r", encoding="utf8") as annotations_io:
            self.index = json.load(annotations_io)

        image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])

        self.image_transforms = transforms.Compose(image_transforms)
        self.id_list = list(self.index.keys())

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        photo_id = self.id_list[idx]
        category = self.index[photo_id]["category"]

        img_path = str(self.root_dir / Path(category) / Path(photo_id + ".jpg"))
        image = Image.open(img_path).convert('RGB')

        if self.image_transforms:
            image = self.image_transforms(image)

        return { "image": image, "txt": self.index[photo_id]["caption"]}


class UnsplashDataModule(pl.LightningDataModule):
    """"
    A lightning compliant datamodule
    """
    unsplash_train = None
    unsplash_val = None
    unsplash_test = None
    unsplash_predict = None

    def __init__(
        self,
        data_dir: str,
        annotations_file: str = "index.json",
        data_transforms: Tuple[Callable, ...] = (),
        train_val_test_split=(0.8, 0.1, 0.1),
        batch_size=32,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.annotations_file = annotations_file
        self.image_transforms = transforms.Compose(data_transforms)
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        unsplash_full = UnsplashDataset(
            self.data_dir, caption_file=self.annotations_file, image_transforms=(self.image_transforms,)
        )
        num_samples = len(unsplash_full)
        train_samples = int(num_samples * self.train_val_test_split[0])
        val_samples = int(num_samples * self.train_val_test_split[1])
        test_samples = num_samples - val_samples - train_samples

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.unsplash_train, self.unsplash_val = random_split(
                unsplash_full, [train_samples, val_samples, test_samples]
            )[:1]

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.unsplash_test = random_split(unsplash_full, [train_samples, val_samples, test_samples])[2]

        if stage == "predict":
            self.unsplash_predict = UnsplashDataset(
                self.data_dir, annotations_file=self.annotations_file, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)
