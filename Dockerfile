FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.11_cuda11.8_opencv

USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    curl \
    gfortran \
    libopenblas-dev \
    liblapack-dev

USER 1000
ENV HOME=/tmp
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Python dependencies
RUN python3 -m pip install --user --upgrade pip setuptools wheel
RUN python3 -m pip install --user --only-binary=:all: numpy==1.26.4 scipy==1.11.4
RUN pip3 install "cython<3.0.0" wheel
RUN pip3 install "pyyaml==5.4.1" --no-build-isolation
RUN pip3 install --user keras==3.11.3 \
                        tensorflow-hub==0.16.1 \
                        opencv-python-headless \
                        tf-models-official==2.19.1 \
                        tf-keras==2.19.0 \
                        tensorflow-model-optimization==0.8.0 \
                        tensorflow[cuda]==2.19.0 \
                        mediapy

RUN pip3 install --user --upgrade tensorflow-datasets==4.8.3

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/tensorflow-models:1.1.1 -f ./Dockerfile  .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/tensorflow-models:1.1.1
