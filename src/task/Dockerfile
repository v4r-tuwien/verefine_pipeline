# use ros as base image
FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04 as nvidia
#FROM ros:kinetic-ros-base

#COPY --from=nvidia /usr/local /usr/local
#COPY --from=nvidia /etc/ld.so.conf.d/glvnd.conf /etc/ld.so.conf.d/glvnd.conf

MAINTAINER Dominik Bauer <bauer@acin.tuwien.ac.at>


# essentials
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils git curl nano tmux unzip wget build-essential cmake python-dev python-pip python-tk

# for opengl/rendering
RUN apt-get install -y libsm6 libglib2.0-0 libgl1-mesa-glx libpcl-dev libglfw3-dev libgles2-mesa-dev \
    libgtk2.0-dev

# -------- install ROS -----------------

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
  python-rosdep \
  python-rosinstall \
  python-vcstools

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# bootstrap rosdep
RUN rosdep init  \
  && rosdep update

# install ros packages
ENV ROS_DISTRO kinetic
RUN apt-get install -y \
  ros-kinetic-ros-base=1.3.2-0*

RUN apt-get update && apt-get install -y --no-install-recommends \
     python-catkin-tools \
     ros-kinetic-tf \
     && rm -rf /var/lib/apt/lists/*

# ----------------

# create a catkin workspace
WORKDIR /task
RUN mkdir -p /task/catkin_ws/src
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd /task/catkin_ws; catkin init'
RUN echo "source /task/catkin_ws/devel/setup.bash" >> /root/.bashrc

# clone and build message and service definitions
RUN /bin/bash -c 'cd /task/catkin_ws/src; \
                  git clone https://github.com/v4r-tuwien/object_detector_msgs.git'
RUN /bin/bash -c 'git clone https://github.com/v4r-tuwien/grasping_pipeline.git'
RUN /bin/bash -c 'git clone https://github.com/markusltnr/sasha_handover.git'
# TODO remove dependencies from grasping and handover (CMake and package.xml)
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd /task/catkin_ws; catkin build'

# clone and build ros_numpy
WORKDIR /task/3rdparty
RUN /bin/bash -c 'cd /task/3rdparty; \
                  git clone https://github.com/eric-wieser/ros_numpy.git'
WORKDIR /task/3rdparty/ros_numpy
RUN python setup.py install

# install python requirements
WORKDIR /task
RUN pip install --upgrade pip==20.3.4
RUN pip install wheel  # smh get errors otherwise
COPY ./src/task/requirements.txt /task/requirements.txt
RUN pip install -r requirements.txt
COPY ./src/util/requirements.txt /task/requirements_renderer.txt
RUN pip install -r requirements_renderer.txt

# install tinyobjloader from source (pypl doesn't work) -- required for rendering of visualizations
WORKDIR /task/3rdparty
RUN git clone https://github.com/syoyo/tinyobjloader.git && \
    cd tinyobjloader/python/ && \
    python setup.py install

# copy src files
WORKDIR /task
COPY ./src/task /task/src/task
COPY ./src/util /task/src/util
COPY ./src/__init__.py /task/src
ENV PYTHONPATH=/task:/task/src:$PYTHONPATH

# copy data for rendering of visualizations
COPY ./data/models.zip /task/data/models.zip
RUN mkdir /verefine
RUN mkdir /verefine/data
RUN unzip /task/data/models.zip -d /verefine/data

# entry point
COPY ./src/densefusion/ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
