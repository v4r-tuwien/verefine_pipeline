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
# desktop full is required since its the only version that has both the  geometry and pcl packages
# TODO: instead of using desktop-full install base ros version and only add required packages
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
                  git clone https://gitlab+deploy-token-28:VxqryrgVYvjaUg4tyJdY@rgit.acin.tuwien.ac.at/msuchi/object_detector_msgs.git'
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd /maskrcnn/catkin_ws; catkin build'

# ----------------

# smh get errors otherwise
RUN pip3 install --upgrade pip
RUN pip3 install wheel setuptools

# clone and build Mask R-CNN (note: replace tensorflow with tensorflow-gpu)
WORKDIR /maskrcnn/3rdparty
RUN /bin/bash -c 'cd /maskrcnn/3rdparty; \
                  git clone https://github.com/matterport/Mask_RCNN.git'

WORKDIR /maskrcnn/3rdparty/Mask_RCNN
RUN sed -i '/tensorflow/ d' requirements.txt && \
    sed -i 's/>=2.0.8/==2.2.4/' requirements.txt && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir tensorflow-gpu==1.12.0
RUN python3 setup.py install

# hack from https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674
RUN pip3 install rospkg catkin_pkg

# install ros_numpy
WORKDIR /maskrcnn/3rdparty
RUN /bin/bash -c 'cd /maskrcnn/3rdparty; \
                  git clone https://github.com/eric-wieser/ros_numpy.git'
WORKDIR /maskrcnn/3rdparty/ros_numpy
RUN python3 setup.py install

# trick from Kiru to get ROS Kinetic to work with Python 3.5
#ENV PYTHONPATH=/usr/local/lib/python3.5/dist-packages:$PYTHONPATH

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
