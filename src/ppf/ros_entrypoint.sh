#!/bin/bash
set -e

# Setup ros environment
source "/opt/ros/melodic/setup.bash"
source "/canister/catkin_ws/devel/setup.bash"
export PYTHONPATH=/ppf/catkin_ws/devel/lib/python3/dist-packages/:$PYTHONPATH

# Load all params
rosparam load /ppf/config/params.yaml locateobject

exec "$@"
