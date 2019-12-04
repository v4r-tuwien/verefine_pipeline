# use ros as base image
FROM ros:kinetic-ros-base

MAINTAINER Dominik Bauer <bauer@acin.tuwien.ac.at>


# essentials
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils git curl nano tmux unzip wget build-essential cmake python-dev python-pip python-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

# create a catkin workspace
WORKDIR /task
RUN mkdir -p /task/catkin_ws/src
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd /task/catkin_ws; catkin init'
RUN echo "source /task/catkin_ws/devel/setup.bash" >> /root/.bashrc

# clone and build message and service definitions
RUN /bin/bash -c 'cd /task/catkin_ws/src; \
                  git clone https://gitlab+deploy-token-28:VxqryrgVYvjaUg4tyJdY@rgit.acin.tuwien.ac.at/msuchi/object_detector_msgs.git'
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
COPY ./src/__init__.py /task/src
ENV PYTHONPATH=/task:/task/src:$PYTHONPATH

# entry point
COPY ./src/densefusion/ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
