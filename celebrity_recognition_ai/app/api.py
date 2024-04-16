# -*- coding: utf-8 -*-
import os

from flask import Flask, jsonify, make_response, request

from celebrity_recognition_ai.app.utils import CelebrityPrediction
from celebrity_recognition_ai.ml.models import CelebrityNet

MODEL_PATH = os.environ["MODEL"]
# CONFIG_PATH = "celebrity_recognition_ai/configs/labels.yaml"
CONFIG_PATH = "celebrity-config.yaml"


model_arch = CelebrityNet(pretrained=False)
predictor = CelebrityPrediction(
    model_arch=model_arch, model_path=MODEL_PATH, config_path=CONFIG_PATH
)

app = Flask(__name__)


@app.route("/celebrity/predict", methods=["POST"])
def prediction_pipeline():
    # Get the image in base 64 and decode it
    payload = request.form.to_dict(flat=False)
    image_b64 = payload["image"][0]
    # Pass it through the inference pipeline
    response = predictor.inference_pipeline(image_b64)
    return make_response(jsonify(response))


@app.route("/celebrity/healthcheck", methods=["GET"])
def healthcheck():
    return "Hello I am well !"


if __name__ == "__main__":
    app.run(debug=False, port=5001, host="0.0.0.0")  # nosec
