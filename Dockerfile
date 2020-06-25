FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3
RUN apt-get update
RUN apt-get install -y python3-grib
COPY ./climate /workspace/climate

