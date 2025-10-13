FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

WORKDIR /app

COPY . .

RUN apt update && apt install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update && apt install -y python3.13 python3.13-dev python3.13-venv

RUN python3.13 -m venv .venv

RUN .venv/bin/pip install --upgrade pip
RUN .venv/bin/pip install --no-cache-dir -r requirements.txt

RUN chmod +x run.sh

CMD ["bash", "run.sh"]

