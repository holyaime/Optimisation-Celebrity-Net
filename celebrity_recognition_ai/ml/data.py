# -*- coding: utf-8 -*-
import os
import random

import albumentations as A  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def get_train_and_validation_images_path(base_path, percentage_train=0.8):

    train_images_absolute_paths = []
    validation_images_absolute_paths = []
    categories = os.listdir(base_path)

    for folder in categories:
        path = os.path.join(base_path, folder)
        # print(folder)
        category_data = os.listdir(path)
        category_images_path = [
            os.path.join(path, element) for element in category_data
        ]
        random.shuffle(category_images_path)
        nb_images_per_category = int(percentage_train * len(category_images_path))
        category_images_path_train = category_images_path[:nb_images_per_category]
        category_images_path_validation = category_images_path[nb_images_per_category:]
        train_images_absolute_paths.extend(category_images_path_train)
        validation_images_absolute_paths.extend(category_images_path_validation)

    return categories, train_images_absolute_paths, validation_images_absolute_paths


class CelebrityDataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, images_absolute_paths, categories, train=True):
        "Initialization"
        self.images_absolute_paths = images_absolute_paths
        self.categories = categories
        self.train = train

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.images_absolute_paths)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        image_path = self.images_absolute_paths[index]
        # Load data and get label
        X = Image.open(image_path)
        X = np.array(X)
        if self.train:
            # Augmentations while training
            self.transforms = A.Compose(
                [
                    A.RandomCrop(width=224, height=224),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                ]
            )
        else:
            self.transforms = A.Compose(
                [A.Resize(width=224, height=224)]  # Achanger plus tard
            )

        X = self.transforms(image=X)["image"]
        X = torch.from_numpy(X).permute(2, 0, 1)
        X = X.float()

        # Labels
        label = image_path.split("/")[-2]
        label = self.categories.index(label)
        y = F.one_hot(torch.tensor(label), num_classes=len(self.categories))
        y = y.float()

        return X, y


# xxxxx
