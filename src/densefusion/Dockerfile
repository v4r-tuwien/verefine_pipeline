# use opengl, cuda and ros as base image
FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04 as nvidia
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
# TODO change for nvidia/cudagl:9.0-devel-ubuntu16.04
FROM ros:kinetic-ros-base

COPY --from=nvidia /usr/local /usr/local
COPY --from=nvidia /etc/ld.so.conf.d/glvnd.conf /etc/ld.so.conf.d/glvnd.conf

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

MAINTAINER Dominik Bauer <bauer@acin.tuwien.ac.at>


# essentials
RUN apt-get update \
    && apt-get install -y --allow-unauthenticated --no-install-recommends \
    apt-utils git curl nano tmux unzip wget build-essential cmake python-dev python-pip python-tk python-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

# for opengl/rendering
RUN apt-get update && apt-get install -y libsm6 libglib2.0-0 libgl1-mesa-glx libpcl-dev libglfw3-dev libgles2-mesa-dev \
    libgtk2.0-dev

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        xz-utils \
        xorg-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev \
        libxxf86vm-dev

# create a catkin workspace
WORKDIR /densefusion
RUN mkdir -p /densefusion/catkin_ws/src
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd /densefusion/catkin_ws; catkin init'
RUN echo "source /densefusion/catkin_ws/devel/setup.bash" >> /root/.bashrc

# clone and build message and service definitions
RUN /bin/bash -c 'cd /densefusion/catkin_ws/src; \
                  git clone https://github.com/v4r-tuwien/object_detector_msgs.git'
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd /densefusion/catkin_ws; catkin build'

# install DenseFusion
WORKDIR /densefusion/3rdparty
RUN git clone https://github.com/j96w/DenseFusion.git
RUN sed -i "s/python3/python/" ./DenseFusion/lib/knn/Makefile # replace first occurance of python3 with python
WORKDIR /densefusion/3rdparty/DenseFusion/lib/knn/build
RUN make ..
RUN touch /densefusion/3rdparty/DenseFusion/lib/__init__.py  # create init file for lib
ENV PYTHONPATH=/densefusion/3rdparty/DenseFusion/lib:/densefusion/3rdparty/DenseFusion/:/verefine/3rdparty/DenseFusion/tools:$PYTHONPATH

# clone and build ros_numpy
WORKDIR /densefusion/3rdparty
RUN /bin/bash -c 'cd /densefusion/3rdparty; \
                  git clone https://github.com/eric-wieser/ros_numpy.git'
WORKDIR /densefusion/3rdparty/ros_numpy
RUN python setup.py install

# install python requirements
WORKDIR /densefusion
RUN pip install --upgrade pip==20.3.4
RUN pip install wheel  # smh get errors otherwise
RUN pip install https://download.pytorch.org/whl/cu100/torchvision-0.4.1%2Bcu100-cp27-cp27mu-linux_x86_64.whl
COPY ./src/densefusion/requirements.txt /densefusion/requirements.txt
RUN pip install -r requirements.txt
COPY ./src/verefine/requirements.txt /densefusion/requirements_verefine.txt
RUN pip install -r requirements_verefine.txt
COPY ./src/util/requirements.txt /densefusion/requirements_renderer.txt
RUN pip install -r requirements_renderer.txt

# install tinyobjloader from source (pypl doesn't work) -- required for rendering in verefine
WORKDIR /task/3rdparty
RUN pip install pybind11
RUN git clone https://github.com/syoyo/tinyobjloader.git && \
    cd tinyobjloader/python/ && \
    python setup.py install

# copy src files
WORKDIR /densefusion
COPY ./src/densefusion /densefusion/src/densefusion
COPY ./src/util /densefusion/src/util
COPY ./src/verefine /densefusion/src/verefine
COPY ./src/__init__.py /densefusion/src
ENV PYTHONPATH=/densefusion:/densefusion/src:$PYTHONPATH

# copy data
COPY ./data /densefusion/data/
# RUN mkdir /densefusion
# RUN mkdir /densefusion/data
RUN mkdir /verefine
RUN unzip /densefusion/data/models.zip -d /verefine/data

# entry point
WORKDIR /densefusion
COPY ./src/densefusion/ros_pose.sh /ros_pose.sh
RUN chmod +x /ros_pose.sh
ENTRYPOINT ["/ros_pose.sh"]
CMD ["bash"]
