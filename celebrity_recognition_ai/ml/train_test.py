# -*- coding: utf-8 -*-
import argparse
import logging
from logging import config

import torch
import yaml
from rich.logging import RichHandler

import data, test_accuracy, logging_config, models
from utils import ParameterError

config.dictConfig(logging_config.logging_config)
logger = logging.getLogger("root")
logger.handlers[0] = RichHandler(markup=True)

training_parser = argparse.ArgumentParser(
    description="Parameters to train a basmatinet model."
)
training_parser.add_argument("datapath", type=str, help="Path to training dataset.")
training_parser.add_argument(
    "--batch-size", type=int, default=16, help="Batch size for training and validation."
)
training_parser.add_argument(
    "--workers", type=int, default=8, help="Number of cpu cores for multiprocessing"
)


training_parser.add_argument("--acceleration", action="store_true", help="If True GPU")

training_parser.add_argument(
    "--config-path",
    type=str,
    default="celebrity_recognition_ai/configs/celebrity-config.yaml",
    help="Path to the yaml file where labels are.",
)

if __name__ == "__main__":
    args = training_parser.parse_args()

    # Defining the parameters
    datapath = args.datapath
    batch_size = args.batch_size
    workers = args.workers
    config_path = args.config_path


    # Setting the device cuda or cpu
    if args.acceleration:
        if torch.cuda.is_available():
            device = torch.device("cuda")

        elif torch.backends.mps.is_available():
            device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Datasets

    with open(config_path, "r") as file:
        content = yaml.safe_load(file)
        categories = content["transverse"]["categories"]

    (train_images_absolute_paths, validation_images_absolute_paths) = (
        data.get_train_and_validation_images_path(
            datapath, categories, percentage_train=0.7
        )
    )


    val_dataset = data.CelebrityDataset(
        validation_images_absolute_paths, categories, train=False
    )

    # Dataloaders
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers
    )

    #Intanciating a model architecture
    net = models.CelebrityNet()
    model_path = '/home/holy/Documents/memoire/celebrity-recognition-ai/celebrity_recognition_ai/app/trained-models/celebritynet.pth'
    net.load_state_dict(torch.load(model_path))

    # #torch.manual_seed(42)
    # net = student_bo.StudentNetbo()
    # model_path = '/home/holy/mes_projets/StudentCelebrity/celebrity-recognition-ai/student_bo_vv1.pth'
    # net.load_state_dict(torch.load(model_path))

    # torch.manual_seed(123)
    # net = mobilenet_v2_100.Mobilenet_100()
    # model_path ='/home/holy/mes_projets/StudentCelebrity/celebrity-recognition-ai/celebrity_recognition_ai/app/trained-models_student/celebritynet2.pth'
    # net.load_state_dict(torch.load(model_path))

    # # Declaring Criterion and Optimizer
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    # net.to(device)

    # Training
    test_accuracy.all_info_validation(
        logger,
        val_dataloader,
        net,
        device
    )
