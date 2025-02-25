FROM ubuntu:22.04
LABEL maintainer="julian.moreno_schneider@dfki.de"

RUN apt-get -y update && \
    apt-get upgrade -y && \
    apt-get update -y &&\
    apt-get install git python3-pip vim -y

RUN python3 -m pip install -U pip
ADD requirements.txt .
RUN pip3 install -r requirements.txt

#RUN apt-get install curl -y && \
#    curl -fsSL https://ollama.com/install.sh | sh

RUN mkdir -p /tmp/sasc

WORKDIR /tmp/sasc/

ADD static static
ADD src/* .
ADD src/models models

ADD script.sh .
RUN chmod -R 777 script.sh

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
EXPOSE 8093

ENTRYPOINT FLASK_APP=flask-rest.py flask run --host=0.0.0.0 --port=8093
#CMD /bin/bash
#CMD ./script.sh
