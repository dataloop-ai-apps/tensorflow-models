FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.10.opencv

USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    curl

USER 1000
ENV HOME=/tmp

RUN pip3 install "cython<3.0.0" wheel
RUN pip3 install "pyyaml==5.4.1" --no-build-isolation
RUN pip3 install --user keras==2.10.0 \
                        tensorflow-hub==0.16.1 \
                        opencv-python-headless \
                        tf-models-official==2.10.1 \
                        mediapy

RUN pip3 install --user --upgrade tensorflow-datasets==4.8.3

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/tensorflow-models:1.1.0 -f ./Dockerfile  .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/tensorflow-models:1.1.0