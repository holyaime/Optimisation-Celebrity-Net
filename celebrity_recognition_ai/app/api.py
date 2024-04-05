# -*- coding: utf-8 -*-
import os

from flask import Flask, redirect, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
app.secret_key = os.urandom(24)


def allowed_file(filename):
    return os.path.splitext(filename)[1] in [".png", ".jpg", ".jpeg"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def render_predict():

    try:
        file = request.files["file"]

        if file:
            redirect("/")

        if file and allowed_file(file.filename):
            return render_template("predict.html")
    except RequestEntityTooLarge:
        return "File is larger than the 5MB limit."

    return redirect("/")


if __name__ == "__main__":
    app.run(debug=False)
