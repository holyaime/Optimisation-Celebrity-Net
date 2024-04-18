# -*- coding: utf-8 -*-
import os
import random

import albumentations as A  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def get_train_and_validation_images_path(
    base_path: str, categories: list[str], percentage_train: float = 0.8
) -> tuple[list[str], list[str]]:

    train_images_absolute_paths: list[str] = []
    validation_images_absolute_paths: list[str] = []

    for folder in categories:
        path = os.path.join(base_path, folder)
        category_data = os.listdir(path)
        category_images_path = []
        for element in category_data:
            image_path = os.path.join(path, element)

            if image_path.endswith(".jpg") or image_path.endswith(".jpg"):
                category_images_path.append(image_path)

        random.shuffle(category_images_path)
        nb_images_per_category = int(percentage_train * len(category_images_path))
        category_images_path_train = category_images_path[:nb_images_per_category]
        category_images_path_validation = category_images_path[nb_images_per_category:]
        train_images_absolute_paths.extend(category_images_path_train)
        validation_images_absolute_paths.extend(category_images_path_validation)

    return train_images_absolute_paths, validation_images_absolute_paths


class CelebrityDataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(
        self,
        images_absolute_paths: list[str],
        categories: list[str],
        train: bool = True,
    ):
        "Initialization"
        self.images_absolute_paths = [
            img_path for img_path in images_absolute_paths if img_path.endswith(".jpg")
        ]
        self.categories = categories
        self.train = train

    def __len__(self) -> int:
        "Denotes the total number of samples"
        return len(self.images_absolute_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        "Generates one sample of data"
        # Select sample
        image_path = self.images_absolute_paths[index]
        # Load data and get label
        X = np.array(Image.open(image_path).convert("RGB"))
        if self.train:
            # Augmentations while training
            self.transforms = A.Compose(
                [
                    A.SmallestMaxSize(max_size=256),
                    A.RandomCrop(width=224, height=224),
                    A.ShiftScaleRotate(
                        shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                    ),
                    A.PixelDropout(p=0.3),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                ]
            )
        else:
            self.transforms = A.Compose(
                [
                    A.SmallestMaxSize(max_size=256),
                    A.CenterCrop(height=224, width=224),
                ]
            )

        X = self.transforms(image=X)["image"]

        X_tensor: torch.Tensor = torch.from_numpy(X).permute(2, 0, 1)
        X_tensor = X_tensor.float()

        # Labels
        label = image_path.split("/")[-2]
        label_index = self.categories.index(label)
        y: torch.Tensor = F.one_hot(
            torch.tensor(label_index), num_classes=len(self.categories)
        )
        y = y.float()
        return X_tensor, y
