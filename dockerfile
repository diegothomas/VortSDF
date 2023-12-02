from nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV CUB_HOME usr/local/cuda-11.0

RUN echo "Installing dependencies..." && \
	apt-get -y --no-install-recommends update && \
	apt-get -y --no-install-recommends upgrade && \
	apt install -y curl

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get install -y \
    cmake \
    libeigen3-dev \
    libgmp-dev \
    libgmpxx4ldbl \
    libmpfr-dev \
    libboost-dev \
    libboost-thread-dev \
    libtbb-dev \
    python3-dev \
    openssh-server \
    build-essential \
    git \
    python3-pip\
    unzip \
    libgl1-mesa-dev #needed


RUN ln /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

RUN git clone https://github.com/PyMesh/PyMesh.git && cd PyMesh && git submodule update --init && pip install -r ./python/requirements.txt && cd third_party && python3 build.py all && cd .. && python3 setup.py build && python3 setup.py install --user && python3 -c "import pymesh; pymesh.test()"

RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pykdtree pyembree # for 11.7

RUN git config --global user.name "diegothomas"
RUN git config --global github.token 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

RUN git clone https://github.com/diegothomas/VortSDF.git && cd VortSDF && pip install -r requirements.txt 

RUN pip install scikit-image human_body_prior
RUN apt install -y python-opengl ninja-build


