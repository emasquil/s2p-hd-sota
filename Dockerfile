#FROM ubuntu:jammy
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
#FROM nvidia/cuda:11.4.3-devel-ubuntu20.04
WORKDIR /home/
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt install -y vim cmake libopencv-dev build-essential gdb libfftw3-dev libgeotiff-dev libtiff5-dev git libgdal-dev python3-pip gdal-bin 
RUN pip3 install cython numba opencv-contrib-python scipy

# just for debug
RUN pip3 install iio fire

# LEGACY S2P for reference should be #3
#RUN git clone https://github.com/centreborelli/s2p.git --recursive && cd s2p && pip3 install -e ".[test]"

RUN pip3 install cffi opencv_python_headless  git+https://github.com/centreborelli/rpcm

# COPY AND INSTALL S2P-HD
COPY ./ /home/s2p-hd/
RUN cd /home/s2p-hd/ &&  pip3 install -e .

#RUN apt-get update &&  apt install -y parallel

RUN useradd -u 1000 user
RUN chown -R 1000:1000 /home/

USER user
