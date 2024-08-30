# -*- coding: utf-8 -*-
import base64
import io

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


class CelebrityPrediction:

    def __init__(self, model_arch, model_path, config_path):
        # Load the model
        self.model_path = model_path
        self.config_path = config_path
        self.model_arch = model_arch

        # Inference image transformations
        self.transforms = A.Compose(
            [
                A.SmallestMaxSize(max_size=256),
                A.CenterCrop(height=224, width=224),
            ]
        )

    @property
    def model(self):
        self.model_arch.load_state_dict(
            torch.load(self.model_path, map_location=torch.device("cpu"))
        )
        self.model_arch.eval()
        return self.model_arch

    @property
    def categories(self):
        with open(self.config_path, "r") as file:
            content = yaml.safe_load(file)
        categories = content["transverse"]["categories"]
        return categories

    def _load_image(self, image_b64):
        image_binary = base64.b64decode(image_b64)
        # Convert the base64 image to PIL Image object
        image_buf = io.BytesIO(image_binary)
        image = Image.open(image_buf).convert("RGB")
        return image

    def _preprocess(self, image):
        X = np.asarray(image)
        X = self.transforms(image=X)["image"]
        X = torch.from_numpy(X).permute(2, 0, 1).unsqueeze(0)
        X = X.float()
        return X

    def _predict(self, X):
        with torch.no_grad():
            out = self.model(X).squeeze(0)
            out = F.softmax(out, dim=-1)
            list_probas = out.numpy()
            list_probas = [str(round(prob, 2)) for prob in list_probas]

        return list_probas

    def _post_process(self, list_probas):

        response = {"categories": self.categories, "probabilities": list_probas}
        return response

    def inference_pipeline(self, image_b64):

        # Load the image
        image = self._load_image(image_b64)
        # Preprocess it
        X = self._preprocess(image)
        # Go through the model and get a prediction
        list_probas = self._predict(X)
        # Post process the prediction and build a response
        response = self._post_process(list_probas)
        return response
