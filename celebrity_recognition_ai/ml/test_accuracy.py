# -*- coding: utf-8 -*-
import logging
import os
from typing import Optional

import mlflow  # type: ignore
import numpy as np
import torch
from models import CelebrityNet
import torch


def one_epoch_validation(
    dataloader,
    model,
    device: torch.device,
) -> float:
    """ """
    model.eval()
    correct = 0.0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
         
            # Get the inputs data and move to device
            images, labels = images.to(device), labels.to(device)
            # Convert one-hot labels to class indices
            class_labels = torch.argmax(labels, dim=1)
            # forward pass
            outputs = model(images)
            #print("labels size",labels.shape)
            # Prediction
        
            _,predicted = torch.max(outputs,1)

           
            # Statistics
            total += labels.size(0)
            print("label_size",labels.size)
            print("label_size(0)",labels.size(0))
            
            # Ensure labels are long tensor
            correct += (predicted == class_labels).sum().item()  # Sum of correct predictions


        average_accuracy = 100*correct/total
    
    return average_accuracy


def all_info_validation(
    logger: logging.Logger,
    val_dataloader: torch.utils.data.DataLoader,
    model: CelebrityNet,
    device: torch.device,
) -> None:
    """ """
    with mlflow.start_run():
  
        # Tracking all parameters in mlflow
        mlflow.log_param("batch_size", val_dataloader.batch_size)

        accuracy = one_epoch_validation(
            val_dataloader, model, device
        )
        # Tracking in mlflow
        mlflow.log_metric("accuracy", accuracy)
        # Print metrics
        logger.info(
            "Accuracy: {} ".format(
                 round(accuracy,8)
                )
            )
    
          
       