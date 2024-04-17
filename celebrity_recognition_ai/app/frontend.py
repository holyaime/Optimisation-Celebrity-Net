# -*- coding: utf-8 -*-
import argparse

import requests

front_parser = argparse.ArgumentParser(
    description="For an input image return the prediction of a dockerized application."
)
front_parser.add_argument(
    "--filename",
    type=str,
    default="../images/arborio.jpg",
    help="Path of the file of which we want the prediction",
)
front_parser.add_argument(
    "--host-ip",
    type=str,
    default="0.0.0.0",  # nosec
    help="IP address of the host of the prediction api",
)
front_parser.add_argument(
    "--port", type=str, default="5001", help="Port for the host of the prediction api"
)

args = front_parser.parse_args()
filename = args.filename
api_url = "http://" + args.host_ip + ":" + args.port + "/celebrity/predict"

with open(filename, "rb") as img:
    img_64 = img.read()

payload = {"image": img_64}
response = requests.post(url=api_url, files=payload)

print("Prediction status code: ", response.status_code)
