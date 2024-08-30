# -*- coding: utf-8 -*-
import torch

from celebrity_recognition_ai.ml import data, engine, models


class TestCelebrityEngine:

    def setup_method(self):

        base_path = "data_test"

        # Other stufs
        self.model = models.CelebrityNet()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.categories = [
            "alassane-dramane-ouattara",
            "arafat-dj",
            "didi-b",
            "didier-drogba",
            "henri-konan-bedie",
            "laurent-gbagbo",
        ]
        self.device = torch.device("cpu")
        (self.train_images_absolute_paths, self.validation_images_absolute_paths) = (
            data.get_train_and_validation_images_path(
                base_path, self.categories, percentage_train=0.8
            )
        )

    def teardown_method(self):
        del self.model
        del self.criterion
        del self.optimizer
        del self.device
        del self.categories
        del self.train_images_absolute_paths
        del self.validation_images_absolute_paths

    def test_one_epoch_training(self):

        # Train Dataset
        train_dataset = data.CelebrityDataset(
            self.train_images_absolute_paths, self.categories, train=True
        )

        # Dataloader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=5, shuffle=True, num_workers=2
        )

        train_loss = engine.one_epoch_training(
            train_dataloader,
            self.model,
            self.criterion,
            self.optimizer,
            self.device,
            breakpoint=2,
        )
        assert train_loss > 0

    def test_one_epoch_validation(self):

        # Validation Dataset
        val_dataset = data.CelebrityDataset(
            self.validation_images_absolute_paths, self.categories, train=False
        )

        # Dataloader
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=5, shuffle=True, num_workers=2
        )

        val_loss = engine.one_epoch_validation(
            val_dataloader, self.model, self.criterion, self.device, breakpoint=2
        )
        assert val_loss > 0
