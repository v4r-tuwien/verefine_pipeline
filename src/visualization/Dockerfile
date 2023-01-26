FROM osrf/ros:melodic-desktop-full-bionic

MAINTAINER Dominik Bauer <bauer@acin.tuwien.ac.at>
LABEL Description="Example pipeline via ROS Melodic (Ubuntu 18.04, Bionic)" Vendor="V4R" Version="1.0"

# --- basics
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils git curl nano tmux unzip wget build-essential cmake python-dev python-pip \
    python3-dev python3-pip python-catkin-tools software-properties-common \
    && rm -rf /var/lib/apt/lists/*
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# --- ROS
ENV ROS_DISTRO melodic
# use Python3 with ROS (hack from https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674)
RUN pip3 install rospkg catkin_pkg
# install ros_numpy
WORKDIR /visualization/3rdparty
RUN /bin/bash -c 'git clone https://github.com/eric-wieser/ros_numpy.git'
WORKDIR /visualization/3rdparty/ros_numpy
RUN python3 setup.py install

# create a catkin workspace
WORKDIR /visualization
COPY ros /visualization/catkin_ws/src
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /visualization/catkin_ws; catkin init'
RUN echo "source /visualization/catkin_ws/devel/setup.bash" >> /root/.bashrc
# build custom message and service definitions
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /visualization/catkin_ws; catkin build'

# --- visualization pipeline
RUN pip3 install numpy Pillow matplotlib
# copy src files
WORKDIR /visualization
COPY visualization /visualization/src

# --- ROS
# entry point
WORKDIR /visualization
COPY visualization/ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
