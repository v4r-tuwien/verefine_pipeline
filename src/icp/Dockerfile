# use cuda and ros as base image
FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04 as nvidia
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
FROM ros:kinetic-ros-base

COPY --from=nvidia /usr/local /usr/local
COPY --from=nvidia /etc/ld.so.conf.d/glvnd.conf /etc/ld.so.conf.d/glvnd.conf

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

MAINTAINER Dominik Bauer <bauer@acin.tuwien.ac.at>


# essentials
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils git curl nano tmux unzip wget build-essential cmake python-dev python-pip python-catkin-tools \
    && rm -rf /var/lib/apt/lists/*


# create a catkin workspace
WORKDIR /icp
RUN mkdir -p /icp/catkin_ws/src
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd /icp/catkin_ws; catkin init'
RUN echo "source /icp/catkin_ws/devel/setup.bash" >> /root/.bashrc

# clone and build message and service definitions
RUN /bin/bash -c 'cd /icp/catkin_ws/src; \
                  git clone https://gitlab+deploy-token-28:VxqryrgVYvjaUg4tyJdY@rgit.acin.tuwien.ac.at/msuchi/object_detector_msgs.git'
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd /icp/catkin_ws; catkin build'

# clone and build ros_numpy
WORKDIR /icp/3rdparty
RUN /bin/bash -c 'cd /icp/3rdparty; \
                  git clone https://github.com/eric-wieser/ros_numpy.git'
WORKDIR /icp/3rdparty/ros_numpy
RUN python setup.py install

# install python requirements
WORKDIR /icp
RUN pip install --upgrade pip
RUN pip install wheel  # smh get errors otherwise
COPY ./src/icp/requirements.txt /icp/requirements.txt
RUN pip install -r requirements.txt

# install tinyobjloader from source (pypl version doesn't seem to work)
WORKDIR /icp/3rdparty
RUN pip install pybind11  # required by setup of tinyobjloader
RUN git clone https://github.com/syoyo/tinyobjloader.git && \
    cd tinyobjloader/python/ && \
    python setup.py install

# copy src files
WORKDIR /icp
COPY ./src/icp /icp/src/icp
COPY ./src/util /icp/src/util
COPY ./src/verefine /icp/src/verefine
COPY ./src/__init__.py /icp/src
ENV PYTHONPATH=/icp:/icp/src:$PYTHONPATH

# copy data
COPY ./data /icp/data/
WORKDIR /icp/data
RUN unzip /icp/data/models.zip
RUN rm /icp/data/models.zip

# entry point
WORKDIR /icp
COPY ./src/icp/ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
