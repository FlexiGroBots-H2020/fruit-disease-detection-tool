#!/bin/bash +x

MODEL_NAME="fruit-model"
INPUT_PATH="@./request.json"
NAMESPACE="common-apps"
SERVICE_HOSTNAME="https://${MODEL_NAME}-predictor-default-${NAMESPACE}.kubeflow.flexigrobots-h2020.eu"
COOKIE="MTY4NDMzNTAyMnxOd3dBTkRWTU5rTkhSa3BFUkVwSlZFNHlOMUJSVVZGTldqVXlUVXBCTTBGV1MwMVdWVk5FUTBZMVNUVTNVVUZLTnpkRVF6SlpURUU9fEbc9o1I2emNrM5NzPvCRIi6WJiRfSZ3HLUpIN9i7jtk"

curl -v ${SERVICE_HOSTNAME}/v1/models/${MODEL_NAME}:predict \
	-d $INPUT_PATH \
	-H "Cookie: authservice_session=${COOKIE}"
