# -*- coding: utf-8 -*-
import logging
import os
from typing import Optional

import mlflow  # type: ignore
import numpy as np
import torch

from models import CelebrityNet


def one_epoch_training(
    dataloader: torch.utils.data.DataLoader,
    model: CelebrityNet,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.SGD,
    device: torch.device,
    breakpoint: Optional[int] = 5,
) -> float:
    """ """
    model.train()
    train_loss = 0.0
    for i, batch in enumerate(dataloader):
        if i == breakpoint:
            break
        # Get the inputs data and move to device
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        # Zero (clear) the parameter gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(images)
        # Compute the loss
        loss = criterion(outputs, labels)
        #  Backward: compute the gradients
        loss.backward()
        #  Optimize: update the weigths
        optimizer.step()
        # Decay LR
        # scheduler.step()
        # Statistics
        train_loss += loss.item()

    return train_loss / len(dataloader)


def one_epoch_validation(
    dataloader: torch.utils.data.DataLoader,
    model: CelebrityNet,
    criterion: torch.nn.CrossEntropyLoss,
    device: torch.device,
    breakpoint: Optional[int] = 4,
) -> float:
    """ """

    model.eval()

    val_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i == breakpoint:
                break
            # Get the inputs data and move to device
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            # forward pass
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            # Statistics
            val_loss += loss.item()

        average_val_loss = val_loss / len(dataloader)
    return average_val_loss


def all_epochs_training_and_validation(
    logger: logging.Logger,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    model: CelebrityNet,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.SGD,
    device: torch.device,
    nb_epochs: int = 20,
    early_stopping: int = 5,
    model_name: str = "trained-model.pth",
    breakpoint: Optional[int] = 2,
) -> None:
    """ """
    with mlflow.start_run():
        counter = 0
        best_val_loss = np.inf
        # Tracking all parameters in mlflow
        mlflow.log_param("nb_epochs", nb_epochs)
        mlflow.log_param("batch_size", train_dataloader.batch_size)
        mlflow.log_param("early_stopping", early_stopping)

        logger.info(
            "Starting a training for {} epochs, with batch size {} and early stopping programmed after {} ".format(
                nb_epochs, train_dataloader.batch_size, early_stopping
            )
        )

        for epoch in range(nb_epochs):
            train_loss = one_epoch_training(
                train_dataloader,
                model,
                criterion,
                optimizer,
                device,
                breakpoint=breakpoint,
            )
            val_loss = one_epoch_validation(
                val_dataloader, model, criterion, device, breakpoint=breakpoint
            )
            # Tracking in mlflow
            mlflow.log_metric("train_loss", train_loss, epoch + 1)
            mlflow.log_metric("val_loss", val_loss, epoch + 1)
            # Print metrics
            logger.info(
                "Epoch:{}, Train Loss: {}  Val Loss: {} ".format(
                    epoch + 1, round(train_loss, 8), round(val_loss, 8)
                )
            )
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # Save the model
                save_path = os.path.join(
                    "celebrity_recognition_ai/app/trained-models/", model_name
                )
                torch.save(model.state_dict(), save_path)
            else:
                counter += 1
            if counter == early_stopping:
                logger.info("======== EARLY STOPPING ========")
                break
