# use ros as base image
#FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04 as nvidia
FROM ros:kinetic-ros-base

#COPY --from=nvidia /usr/local /usr/local
#COPY --from=nvidia /etc/ld.so.conf.d/glvnd.conf /etc/ld.so.conf.d/glvnd.conf

MAINTAINER Dominik Bauer <bauer@acin.tuwien.ac.at>


# essentials
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils git curl nano tmux unzip wget build-essential cmake python-dev python-pip python-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

## for opengl/rendering
#RUN apt-get update && apt-get install -y libsm6 libglib2.0-0 libgl1-mesa-glx libpcl-dev libglfw3-dev libgles2-mesa-dev

# create a catkin workspace
WORKDIR /task
RUN mkdir -p /task/catkin_ws/src
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd /task/catkin_ws; catkin init'
RUN echo "source /task/catkin_ws/devel/setup.bash" >> /root/.bashrc

# clone and build message and service definitions
RUN /bin/bash -c 'cd /task/catkin_ws/src; \
                  git clone https://gitlab+deploy-token-28:VxqryrgVYvjaUg4tyJdY@rgit.acin.tuwien.ac.at/msuchi/object_detector_msgs.git'
RUN /bin/bash -c 'git clone https://gitlab+deploy-token-29:_3RcqWbSB6SHJz3xA2Bh@rgit.acin.tuwien.ac.at/leitnermrks/grasping_pipeline.git'
RUN /bin/bash -c 'git clone https://gitlab+deploy-token-30:eKjhFEKP9g4iyqdms3Ye@rgit.acin.tuwien.ac.at/demo/sasha_handover.git'
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd /task/catkin_ws; catkin build'

# clone and build ros_numpy
WORKDIR /task/3rdparty
RUN /bin/bash -c 'cd /task/3rdparty; \
                  git clone https://github.com/eric-wieser/ros_numpy.git'
WORKDIR /task/3rdparty/ros_numpy
RUN python setup.py install

# install python requirements
WORKDIR /task
RUN pip install --upgrade pip
RUN pip install wheel  # smh get errors otherwise
COPY ./src/task/requirements.txt /task/requirements.txt
RUN pip install -r requirements.txt

# copy src files
WORKDIR /task
COPY ./src/task /task/src/task
#COPY ./src/util /task/src/util
COPY ./src/__init__.py /task/src
ENV PYTHONPATH=/task:/task/src:$PYTHONPATH

# entry point
COPY ./src/densefusion/ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
