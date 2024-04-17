# -*- coding: utf-8 -*-
import pytest
import torch

from celebrity_recognition_ai.ml import models


@pytest.fixture
def model():
    return models.CelebrityNet()


@pytest.mark.parametrize(
    "batch_size, execepted_type", [(5, torch.Tensor), (10, torch.Tensor)]
)
def test_models(model, batch_size, execepted_type):
    """
    We check that our models returns effectively an torch.Tensor object
    """
    img = torch.randn(size=(batch_size, 3, 224, 224))

    assert type(model(img)) == execepted_type
