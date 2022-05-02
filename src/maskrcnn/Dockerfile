FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

MAINTAINER Dominik Bauer <bauer@acin.tuwien.ac.at>

# essentials
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils git curl nano tmux unzip wget build-essential cmake python-dev python-pip \
    python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# -------- install ROS -----------------

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
  python-rosdep \
  python-rosinstall \
  python-vcstools \
  && rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# bootstrap rosdep
RUN rosdep init  \
  && rosdep update

# install ros packages
ENV ROS_DISTRO kinetic
RUN apt-get update && apt-get install -y \
  ros-kinetic-ros-base=1.3.2-0* \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
     python-catkin-tools

# create a catkin workspace
WORKDIR /maskrcnn
RUN mkdir -p /maskrcnn/catkin_ws/src
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd /maskrcnn/catkin_ws; catkin init'
RUN echo "source /maskrcnn/catkin_ws/devel/setup.bash" >> /root/.bashrc

# clone and build message and service definitions
RUN /bin/bash -c 'cd /maskrcnn/catkin_ws/src; \
                  git clone https://github.com/v4r-tuwien/object_detector_msgs.git'
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd /maskrcnn/catkin_ws; catkin build'

# ----------------

# smh get errors otherwise
RUN pip3 install --upgrade wheel setuptools==51.0.0 pip==20.3.4


# clone and build Mask R-CNN (note: replace tensorflow with tensorflow-gpu)
WORKDIR /maskrcnn/3rdparty
RUN /bin/bash -c 'cd /maskrcnn/3rdparty; \
                  git clone https://github.com/matterport/Mask_RCNN.git'

WORKDIR /maskrcnn/3rdparty/Mask_RCNN
RUN pip3 install --no-cache-dir numpy==1.18.5 scipy==1.4.1 Pillow==7.2.0 cython==0.29.28 matplotlib==3.0.3 scikit-image==0.15.0 tensorflow-gpu==1.12.0 keras==2.2.4 opencv-python==4.4.0.42 h5py==2.10.0 imgaug==0.4.0 IPython==7.9.0
RUN python3 setup.py install

# hack from https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674
RUN pip3 install rospkg catkin_pkg

# install ros_numpy
WORKDIR /maskrcnn/3rdparty
RUN /bin/bash -c 'cd /maskrcnn/3rdparty; \
                  git clone https://github.com/eric-wieser/ros_numpy.git'
WORKDIR /maskrcnn/3rdparty/ros_numpy
RUN python3 setup.py install

# copy src files
WORKDIR /maskrcnn
COPY ./src/maskrcnn /maskrcnn/src/maskrcnn
COPY ./src/util /maskrcnn/src/util
COPY ./src/__init__.py /maskrcnn/src
ENV PYTHONPATH=/maskrcnn:/maskrcnn/src:$PYTHONPATH

# copy data
COPY ./data /maskrcnn/data/
WORKDIR /maskrcnn/data

# entry point
WORKDIR /maskrcnn
COPY ./src/maskrcnn/ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
