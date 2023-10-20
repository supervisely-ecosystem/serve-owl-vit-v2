FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ENV PYTHONUNBUFFERED=1 
RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    build-essential apt-utils \
    wget curl vim git ca-certificates kmod \
    nvidia-driver-525 \
    && rm -rf /var/lib/apt/lists/*
RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update --yes --quiet
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-lib2to3 \
    python3.10-gdbm \
    python3.10-tk \
    pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 999 \
    && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip
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

RUN python3 -m pip install -U scikit-image