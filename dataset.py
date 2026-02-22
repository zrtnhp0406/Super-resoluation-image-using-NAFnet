import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torch


class SRDataset(Dataset):
    def __init__(
        self,
        root_dir,
        hr_size=(32, 64),
        scale=2,
        augment=True
    ):
        """
        hr_size: patch HR (H, W)
        scale: SR scale (2, 4, ...)
        augment: random flip
        """

        self.hr_paths = []
        self.hr_size = hr_size
        self.scale = scale
        self.augment = augment

        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().startswith("hr-") and f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.hr_paths.append(os.path.join(root, f))

        self.hr_paths = sorted(self.hr_paths)
        self.to_tensor = T.ToTensor()

        print(f"Found {len(self.hr_paths)} HR images.")

    def __len__(self):
        return len(self.hr_paths)

    def random_crop(self, img):
        w, h = img.size
        th, tw = self.hr_size

        if w == tw and h == th:
            return img

        if w < tw or h < th:
            img = TF.resize(img, (max(h, th), max(w, tw)), InterpolationMode.BICUBIC)
            w, h = img.size

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return TF.crop(img, i, j, th, tw)

    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]

        try:
            hr = Image.open(hr_path).convert("RGB")
        except:
            return self.__getitem__(random.randint(0, len(self)-1))

        # ---- Random crop HR patch ----
        hr = self.random_crop(hr)

        # ---- Augmentation ----
        if self.augment:
            if random.random() < 0.5:
                hr = TF.hflip(hr)
            if random.random() < 0.5:
                hr = TF.vflip(hr)

        # ---- Create LR ----
        lr_size = (
            self.hr_size[0] // self.scale,
            self.hr_size[1] // self.scale
        )

        lr = TF.resize(
            hr,
            lr_size,
            interpolation=InterpolationMode.BICUBIC
        )

        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        return lr, hr
