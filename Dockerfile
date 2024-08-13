FROM docker.io/dataloopai/dtlpy-agent:cpu.py3.10.opencv
USER root
RUN apt update && apt install -y curl gpg software-properties-common

USER 1000
WORKDIR /tmp
ENV HOME=/tmp
RUN pip install \
    tf-models-official \
    mediapy
RUN pip install --upgrade tensorflow-datasets==4.8.3
RUN sudo apt update
RUN sudo apt install ffmpeg

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/tensorflow-models:0.0.1 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/tensorflow-models:0.0.1

# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/tensorflow-models:0.0.1 bash