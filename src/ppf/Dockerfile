FROM ros:melodic-perception-bionic

MAINTAINER Dominik Bauer <bauer@acin.tuwien.ac.at>
LABEL Description="LocateObject - via ROS Melodic (Ubuntu 18.04, Bionic)" Vendor="V4R" Version="1.0"

# --- basics
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    apt-transport-https \
    git \
    curl \
    nano \
    tmux \
    unzip \
    wget \
    build-essential \
    cmake \
    libcmph-dev \
    python3-dev \
    python3-pip \
    python3-catkin-pkg-modules \
    python3-rospkg-modules \
    python3-empy \
    python-catkin-tools \
    software-properties-common \
    ros-melodic-vision-msgs \
    && rm -rf /var/lib/apt/lists/*
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# --- OpenGL (via https://gitlab.com/nvidia/container-images/opengl/-/blob/ubuntu18.04/
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglvnd0 libgl1 libglx0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=nvidia/opengl:1.0-glvnd-runtime-ubuntu18.04 /usr/share/glvnd/egl_vendor.d/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# -- CUDA
COPY --from=nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 /usr/local/cuda /usr/local/cuda
COPY --from=nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/lib/x86_64-linux-gnu/libcudnn.so
ENV PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all


# --- ROS
ENV ROS_DISTRO melodic
# use Python3 with ROS (hack from https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca3>
RUN pip3 install rospkg catkin_pkg
# install ros_numpy
WORKDIR /ppf/3rdparty
RUN /bin/bash -c 'git clone https://github.com/eric-wieser/ros_numpy.git'
WORKDIR /ppf/3rdparty/ros_numpy
RUN python3 setup.py install

# create a python3 catkin workspace
WORKDIR /ppf
COPY src/ppf/src/tracebot_msgs /ppf/catkin_ws/src/tracebot_msgs
RUN /bin/bash -c 'cd /ppf/catkin_ws/src; \
                  git clone https://github.com/v4r-tuwien/object_detector_msgs.git'
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /ppf/catkin_ws; catkin init; catkin build'
RUN echo "source /ppf/catkin_ws/devel/setup.bash" >> /root/.bashrc
RUN /bin/bash -c 'cd /ppf/catkin_ws/; apt-get update; source /ppf/catkin_ws/devel/setup.bash; \
                  wstool init; wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.5; wstool up; \
                  rosdep install --from-paths src --ignore-src -y -r; rm -rf /var/lib/apt/lists/*'
# build custom message and service definitions
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /ppf/catkin_ws; catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so'

# --- LOCATE OBJECT DEPENDENCIES
# update cmake (via KitWare)
RUN /bin/bash -c "apt-get update;\
                  wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null; \
                  echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null; \
                  apt-get update && apt-get install -y cmake && rm -rf /var/lib/apt/lists/*"
# build pybind11 from source
RUN pip3 install pytest
WORKDIR /ppf/3rdparty
RUN /bin/bash -c 'git clone https://github.com/pybind/pybind11.git'
WORKDIR /ppf/3rdparty/pybind11/build
RUN /bin/bash -c 'cmake .. && make -j$(getconf _NPROCESSORS_ONLN) && make install && rm -rf *'
# build glog from source
WORKDIR /ppf/3rdparty
RUN /bin/bash -c 'git clone https://github.com/google/glog.git'
WORKDIR /ppf/3rdparty/glog/build
RUN /bin/bash -c 'cmake .. && make -j$(getconf _NPROCESSORS_ONLN) && make install && rm -rf *'

# --- LOCATE OBJECT
# build PPF project with python bindings
COPY src/ppf/src/ppf /ppf/src/ppf
WORKDIR /ppf/src/ppf/build
RUN /bin/bash -c 'cmake .. -DPYBIND11_PYTHON_VERSION=3.6 && make -j$(getconf _NPROCESSORS_ONLN) && make install'
# plane pop-out and canister demo dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --ignore-installed PyYAML numpy==1.19.1 imageio==2.9.0 scikit-image==0.17.2 scipy==1.5.2 open3d==0.15.2
RUN pip3 install PyOpenGL-accelerate==3.1.6
# verification dependencies
RUN pip3 install Cython  # needs to be installed first
RUN pip3 install trimesh==3.9.9 pyopengl==3.1.5 pyglet==1.5.15 glumpy==1.1.0 opencv-python-headless==4.5.1.48 pybullet==2.7.5

COPY src/ppf/src/locate_object /ppf/src/locate_object
COPY src/ppf/src/verefine /ppf/src/verefine
COPY src/ppf/src/utility /ppf/src/utility
COPY src/ppf/config /ppf/config
RUN mkdir -p /verefine/data
COPY ./data /verefine/data
COPY ./src/verefine/util /ppf/src/locate_object/util

RUN mkdir -p /config
COPY ./config /config

# --- CLEANUP
# RUN rm -rf /var/lib/apt/lists/*

# --- ROS
# entry point
WORKDIR /ppf
COPY src/ppf/ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
