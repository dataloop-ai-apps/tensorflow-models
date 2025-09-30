FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.10.opencv

USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    curl

# CUDA Toolkit 11.8 for TensorFlow XLA support
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y cuda

USER 1000
ENV HOME=/tmp
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda/nvvm/libdevice

# Python dependencies
RUN pip3 install "cython<3.0.0" wheel
RUN pip3 install "pyyaml==5.4.1" --no-build-isolation
RUN pip3 install --user keras==3.11.3 \
                        tensorflow-hub==0.16.1 \
                        opencv-python-headless \
                        tf-models-official==2.10.1 \
                        tensorflow[cuda]==2.20.0 \
                        mediapy

RUN pip3 install --user --upgrade tensorflow-datasets==4.8.3

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/tensorflow-models:1.1.1 -f ./Dockerfile  .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/tensorflow-models:1.1.1