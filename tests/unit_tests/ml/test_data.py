# -*- coding: utf-8 -*-
import pytest
import torch
import yaml

from celebrity_recognition_ai.ml import data

@pytest.fixture
def categorie():
    with open("celebrity_recognition_ai/configs/celebrity-config.yaml" ,"r") as file:
        content = yaml.safe_load(file)
        categories = content["transverse"]["categories"]
    return categories

@pytest.fixture
def resources(categorie):
    return data.get_train_and_validation_images_path("./data_test", categorie)


def test_train_validation_length(resources):

    expected_data_length = 42

    train_images_absolute_paths, validation_image_absolute_path = resources
    # test origin data length with train and validation length
    assert expected_data_length == len(train_images_absolute_paths) + len(validation_image_absolute_path )


def test_ouput_image(resources, categorie):

    expected_image = torch.Size([3, 224, 224])
    expected_categorie = 6
    (train_images_absolute_paths,_) = resources

    train_loader = data.CelebrityDataset(train_images_absolute_paths, categorie)

    output_image,output_categorie = train_loader[1]

    # test output_image shape

    assert expected_image == output_image.shape

    # test output categorie

    assert expected_categorie == output_categorie.shape[0]
