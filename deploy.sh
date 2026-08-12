#!/bin/bash
# Simple deploy script (example) to run the docker image locally
IMAGE_NAME=dzpoker_demo:latest

docker build -t $IMAGE_NAME .
docker run -p 8000:8000 $IMAGE_NAME
