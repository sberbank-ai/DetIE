FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

WORKDIR ~/build

RUN chmod 1777 /tmp
RUN pip install --upgrade pip cmake
RUN apt update --allow-unauthenticated && apt install -y build-essential

RUN pip install lapsolver==1.1.0
COPY ./context/requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

RUN rm -rf ~/build

COPY . /project
WORKDIR /project

