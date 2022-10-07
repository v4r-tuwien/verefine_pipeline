FROM ros:melodic-perception-bionic

# install packages
RUN apt-get update && apt-get install -q -y \
  dirmngr \
  gnupg2 \
#   lsb-release \
# install packages for building software
#   cmake \
  git \
# some more needed packages
  sudo \
  wget \
# catkin build >> catkin_make
  python-catkin-tools \
# some useful programs
  vim \
  tmux \
# dependencies for ppf_recognizer
  libcmph-dev \
  # libgoogle-glog-dev \  # glog package doesn't include cmake config scripts ??
# needed for pybind11
  python3-dev \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

# create a user
RUN adduser --gecos "ROS User" --disabled-password v4r

# remove the need to enter a sudo password for the v4r user
RUN echo "v4r ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER v4r

# Set HOME evnironment variable
RUN HOME=/home/v4r

# build and install google glog
RUN /bin/bash -c 'cd /home/v4r; git clone --branch v0.4.0 https://github.com/google/glog.git; \
                  cd glog; cmake -H. -Bbuild -G "Unix Makefiles"; cmake --build build; \
                  sudo cmake --build build --target install; cd ..; sudo rm -rf glog'

# build and install pybind11,
# using the pybind11-dev package causes a strange error when building the pyppf target
RUN /bin/bash -c 'python3 -m pip install pytest'
RUN /bin/bash -c 'cd /home/v4r/; git clone --branch v2.4.3 https://github.com/pybind/pybind11.git; cd pybind11; \
                  mkdir build; cd build; cmake ..; make -j$(nproc); sudo make install; \
                  cd ../..; sudo rm -rf pybind11'

# clone ppf repo
COPY /src/ppf /home/v4r/ppf
#RUN /bin/bash -c 'cd /home/v4r; git clone https://gitlab+deploy-token-24:_QgAnkQE-5yrTLP7woPV@rgit.acin.tuwien.ac.at/jeremy.itamah/ppf.git; cd ppf; git checkout 79c647c6'
RUN /bin/bash -c 'cd /home/v4r/ppf; sudo mkdir build; cd build; sudo cmake ..; sudo make -j$(nproc)'
# RUN /bin/bash -c 'cd home/v4r; rm -rf ppf' # uncomment once ppf repo supports installing files

# create a catkin workspace
RUN mkdir -p /home/v4r/catkin_ws/src
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /home/v4r/catkin_ws; catkin init'
RUN echo "source /home/v4r/catkin_ws/devel/setup.bash" >> /home/v4r/.bashrc

# clone ros packages, gitlab deploy tokens are needed since the repos aren't public
COPY /src/ppf_recognizer_ros /home/v4r/catkin_ws/src/ppf_recognizer_ros
#RUN /bin/bash -c 'cd /home/v4r/catkin_ws/src; \
#                  git clone https://gitlab+deploy-token-22:zKCM9FnYr86bgaABY63_@rgit.acin.tuwien.ac.at/jeremy.itamah/ppf_recognizer_ros.git; \
#                  cd ppf_recognizer_ros; git checkout bbf03129'

COPY /src/ppf_recognizer_ros_msgs /home/v4r/catkin_ws/src/ppf_recognizer_ros_msgs
#RUN /bin/bash -c 'cd /home/v4r/catkin_ws/src; \
#                  git clone https://gitlab+deploy-token-23:xzQLssogTa-t5igeXrv8@rgit.acin.tuwien.ac.at/jeremy.itamah/ppf_recognizer_ros_msgs.git; \
 #                 cd ppf_recognizer_ros_msgs; git checkout 05206cc5'
#
RUN /bin/bash -c 'cd /home/v4r/catkin_ws/src; \
                  git clone https://github.com/v4r-tuwien/object_detector_msgs.git;'

RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /home/v4r/catkin_ws; catkin build -DPPFRecognizer_DIR=/home/v4r/ppf/build -DCMAKE_MODULE_PATH="/home/v4r/ppf/cmake;${CMAKE_MODULE_PATH}"'

RUN /bin/bash -c 'cd /home/v4r/catkin_ws/src/ppf_recognizer_ros; sudo touch CATKIN_IGNORE;'

## setup entrypoint
COPY ./ros_entrypoint.sh /
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
