FROM osrf/ros:melodic-desktop-full-bionic

MAINTAINER Dominik Bauer <bauer@acin.tuwien.ac.at>
LABEL Description="VeREFINE with DenseFusion via ROS Melodic (Ubuntu 18.04, Bionic)" Vendor="V4R" Version="1.0"

# --- basics
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils git curl nano tmux unzip wget build-essential cmake python-dev python-pip \
    python3-dev python3-pip python-catkin-tools software-properties-common \
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
# use Python3 with ROS (hack from https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674)
RUN pip3 install --upgrade setuptools
RUN pip3 install rospkg catkin_pkg
# install ros_numpy
WORKDIR /verefine/3rdparty
RUN /bin/bash -c 'git clone https://github.com/eric-wieser/ros_numpy.git'
WORKDIR /verefine/3rdparty/ros_numpy
RUN python3 setup.py install

# create a catkin workspace
WORKDIR /verefine
RUN mkdir -p /verefine/catkin_ws/src
RUN /bin/bash -c 'cd /verefine/catkin_ws/src; \
                  git clone https://github.com/v4r-tuwien/object_detector_msgs.git'
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /verefine/catkin_ws; catkin init'
RUN echo "source /verefine/catkin_ws/devel/setup.bash" >> /root/.bashrc
# build custom message and service definitions
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /verefine/catkin_ws; catkin build'

# --- VeREFINE pipeline
# clone verefine
WORKDIR /verefine/3rdparty
RUN git clone https://github.com/dornik/verefine.git
ENV PYTHONPATH=/verefine/3rdparty/verefine:/verefine/3rdparty/verefine/src:$PYTHONPATH
WORKDIR /verefine
COPY src/verefine/requirements.txt /verefine/
RUN pip3 install --upgrade pip
RUN pip3 install setuptools wheel numpy cython scikit-build
RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade sklearn open3d

# clone DenseFusion
WORKDIR /verefine/3rdparty/verefine/3rdparty
RUN git clone https://github.com/j96w/DenseFusion.git
RUN ln -s /verefine/3rdparty/verefine/3rdparty/DenseFusion /verefine/3rdparty/
RUN touch /verefine/3rdparty/DenseFusion/lib/__init__.py  # create init file for lib
ENV PYTHONPATH=/verefine/3rdparty/DenseFusion/lib:/verefine/3rdparty/DenseFusion/:/verefine/3rdparty/DenseFusion/tools:$PYTHONPATH

# copy additional src files (ROS wrapper)
WORKDIR /verefine/
COPY src/verefine /verefine/src
COPY data/densefusion_* /verefine/data/
COPY data/models /verefine/data/models
COPY data/ycbv_* /verefine/data/

RUN mkdir -p /config
COPY ./config /config

# --- ROS
# entry point
WORKDIR /verefine
COPY src/verefine/ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
