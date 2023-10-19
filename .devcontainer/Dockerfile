FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt install python3-pip -y
RUN apt-get install -y git
RUN python3 -m pip install --upgrade pip
RUN pip install cython

RUN git clone https://github.com/google-research/scenic.git
WORKDIR /scenic
RUN python3 -m pip install -vq .
RUN python3 -m pip install -r scenic/projects/owl_vit/requirements.txt
WORKDIR /
RUN mkdir /big_vision
RUN git clone https://github.com/google-research/big_vision.git /big_vision
RUN python3 -m pip install -r /big_vision/big_vision/requirements.txt


RUN python3 -m pip install supervisely==6.72.156
RUN apt-get install ffmpeg libgeos-dev libsm6 libxext6 libexiv2-dev libxrender-dev libboost-all-dev -y

RUN apt-get -y install curl
RUN apt-get -y install wget
RUN apt -y install wireguard iproute2
