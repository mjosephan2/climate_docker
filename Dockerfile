FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3
RUN apt-get update
RUN apt-get install -y python3-grib
RUN pip install sklearn
COPY ./climate /workspace/climate

# nvidia-docker run --name=climate_model \
#                   -v /home/clima_modelers/data:/workspace/climate/data \
#                   -v /home/clima_modelers/model:/workspace/climate/model \
#                   -it climate

# nvidia-docker build . -t climate