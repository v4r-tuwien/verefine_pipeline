# use cuda and ros as base image
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
FROM ros:kinetic-ros-base

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

MAINTAINER Dominik Bauer <bauer@acin.tuwien.ac.at>


# essentials
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils git curl nano tmux unzip wget build-essential cmake python-dev python-pip python-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

# create a catkin workspace
WORKDIR /densefusion
RUN mkdir -p /densefusion/catkin_ws/src
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd /densefusion/catkin_ws; catkin init'
RUN echo "source /densefusion/catkin_ws/devel/setup.bash" >> /root/.bashrc

# clone and build message and service definitions
RUN /bin/bash -c 'cd /densefusion/catkin_ws/src; \
                  git clone https://gitlab+deploy-token-28:VxqryrgVYvjaUg4tyJdY@rgit.acin.tuwien.ac.at/msuchi/object_detector_msgs.git'
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
RUN pip install --upgrade pip
RUN pip install wheel  # smh get errors otherwise
COPY ./src/densefusion/requirements.txt /densefusion/requirements.txt
RUN pip install -r requirements.txt

# copy src files
WORKDIR /densefusion
COPY ./src/densefusion /densefusion/src/densefusion
COPY ./src/util /densefusion/src/util
COPY ./src/verefine /densefusion/src/verefine
COPY ./src/__init__.py /densefusion/src
ENV PYTHONPATH=/densefusion:/densefusion/src:$PYTHONPATH

# copy data
COPY ./data /densefusion/data/
#RUN unzip /densefusion/data/models.zip

# entry point
COPY ./src/densefusion/ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
