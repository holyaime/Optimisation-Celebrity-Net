# -*- coding: utf-8 -*-
import base64
import os

from flask import Flask, render_template, request

from celebrity_recognition_ai.app.utils import CelebrityPrediction
from celebrity_recognition_ai.ml.models import CelebrityNet

MODEL_PATH = os.environ["MODEL"]
CONFIG_PATH = "celebrity-config.yaml"


model_arch = CelebrityNet(pretrained=False)
predictor = CelebrityPrediction(
    model_arch=model_arch, model_path=MODEL_PATH, config_path=CONFIG_PATH
)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/celebrity/predict", methods=["POST"])
def prediction_pipeline():
    # Retrieve image
    img = request.files["image"]

    # Encode to base64
    img_b64 = base64.b64encode(img.read())

    # Pass it trough model
    response = predictor.inference_pipeline(img_b64)
    response["probabilities"] = [
        100 * (float(p) + 0.005) for p in response["probabilities"]
    ]

    return render_template("predict.html", predicts=response)


@app.route("/celebrity/healthcheck", methods=["GET"])
def healthcheck():
    return "Hello I am well !"


if __name__ == "__main__":
    app.run(debug=True, port=5001, host="0.0.0.0")  # nosec
