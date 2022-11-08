FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04

# magic to stop tzdata from asking timezone 
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing -y
RUN apt install python3 -y
RUN apt-get install git-all -y
RUN apt install python3-pip -y

RUN pip install numpy 
RUN pip install torch
RUN pip install torchvision
RUN pip install torchaudio

RUN apt-get install libsndfile1 -y
RUN pip install pyannote.audio

RUN pip install git+https://github.com/openai/whisper.git 
RUN apt update -y && apt install ffmpeg -y
RUN pip install setuptools-rust

# test imports and download models 
RUN python3 -c "import pyannote.audio" 
RUN python3 -c "import whisper; whisper.load_model(\"base\")" 

RUN pip install fastapi
RUN pip install "uvicorn[standard]"

WORKDIR ara
COPY sample_data sample_data
COPY output output
COPY main.py main.py
COPY src src

CMD uvicorn main:app --host 0.0.0.0 --port 80
