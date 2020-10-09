FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN useradd -m achraf

RUN chown -R achraf:achraf /home/achraf/

COPY --chown=achraf . /home/achraf/app/

USER achraf

RUN cd /home/achraf/app/ && pip3 install --upgrade pip && pip3 install -r requirements.txt

WORKDIR /home/achraf/app