# -*- coding: utf-8 -*-
import logging
import os
from unittest.mock import patch

import pytest
import torch
from rich.logging import RichHandler

from celebrity_recognition_ai.ml import data, engine, logging_config, models


@pytest.fixture
def epoch_args():
    logging.config.dictConfig(logging_config.logging_config)
    logger = logging.getLogger("root")
    logger.handlers[0] = RichHandler(markup=True)

    model = models.CelebrityNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    categories = [
        "alassane-dramane-ouattara",
        "arafat-dj",
        "didi-b",
        "didier-drogba",
        "henri-konan-bedie",
        "laurent-gbagbo",
    ]
    device = torch.device("cpu")
    (train_images_absolute_paths, validation_images_absolute_paths) = (
        data.get_train_and_validation_images_path(
            "data_test", categories, percentage_train=0.8
        )
    )

    # Train Dataset
    train_dataset = data.CelebrityDataset(
        train_images_absolute_paths, categories, train=True
    )

    # Validation Dataset
    val_dataset = data.CelebrityDataset(
        validation_images_absolute_paths, categories, train=False
    )

    # Dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=5, shuffle=True, num_workers=2
    )

    # Dataloader
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=5, shuffle=True, num_workers=2
    )

    yield logger, train_dataloader, val_dataloader, model, criterion, optimizer, device

    if os.path.isfile("celebrity_recognition_ai/app/trained-models/celebrity_fake.pth"):
        os.remove("celebrity_recognition_ai/app/trained-models/celebrity_fake.pth")

    if os.path.isfile("celebrity_recognition_ai/app/trained-models/fake_celebrity.pth"):
        os.remove("celebrity_recognition_ai/app/trained-models/fake_celebrity.pth")


def test_is_model_saved(epoch_args):

    logger, train_dataloader, val_dataloader, model, criterion, optimizer, device = (
        epoch_args
    )

    model.to(device)

    engine.all_epochs_training_and_validation(
        logger,
        train_dataloader,
        val_dataloader,
        model,
        criterion,
        optimizer,
        device,
        nb_epochs=10,
        early_stopping=3,
        model_name="celebrity_fake.pth",
        breakpoint=2,
    )

    assert os.path.isfile(
        "celebrity_recognition_ai/app/trained-models/celebrity_fake.pth"
    )


@pytest.mark.parametrize(
    "val_loss_history, early_stopping, expected_early_stopping",
    [
        ([0.5, 0.4, 0.35, 0.34, 0.36, 0.38, 0.4], 3, 3),
        (
            [
                1.19719442,
                1.17974965,
                1.20041863,
                1.19467191,
                1.18788079,
                1.20467957,
                1.19966793,
            ],
            5,
            5,
        ),
    ],
)
def test_early_stopping(
    epoch_args, val_loss_history, early_stopping, expected_early_stopping
):

    logger, train_dataloader, val_dataloader, model, criterion, optimizer, device = (
        epoch_args
    )

    with patch(
        "celebrity_recognition_ai.ml.engine.one_epoch_training"
    ) as mock_training:
        mock_training.side_effect = [0.1] * len(
            val_loss_history
        )  # Simuler une perte d'entraînement constante
        with patch(
            "celebrity_recognition_ai.ml.engine.one_epoch_validation"
        ) as mock_validation:
            mock_validation.side_effect = val_loss_history  # Utiliser val_loss_history comme métriques de validation simulées
            engine.all_epochs_training_and_validation(
                logger,
                train_dataloader,
                val_dataloader,
                model,
                criterion,
                optimizer,
                device,
                nb_epochs=15,
                early_stopping=early_stopping,
                model_name="fake_celebrity.pth",
                breakpoint=2,
            )

    # Vérifier si l'arrêt anticipé est respecté
    best_loss = min(val_loss_history)
    index_best_loss = val_loss_history.index(best_loss)
    i = index_best_loss + 1
    assert len(val_loss_history[i:]) == expected_early_stopping
