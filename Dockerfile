FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.10.opencv

USER root
RUN apt update && apt install -y curl

USER 1000
ENV HOME=/tmp
RUN python3 -m pip install --upgrade pip
COPY /requirements.txt .

RUN pip install -r requirements.txt
RUN pip install --upgrade tensorflow-datasets==4.8.3
RUN pip install tensorflow==2.10.1 keras==2.10.0 tensorflow-hub==0.16.1

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/tensorflow-models:0.0.1 -f ./Dockerfile  .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/tensorflow-models:0.0.1