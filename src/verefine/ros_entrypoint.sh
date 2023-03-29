#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/melodic/setup.bash"
source "/verefine/catkin_ws/devel/setup.bash"

# start the detect_objects service in a tmux session
tmux new-session -d -s "verefine"
tmux selectp -t 0
tmux send-keys "ROS_NAMESPACE=verefine python3 /verefine/src/ros_estimate_poses.py" C-m
tmux attach-session -t "verefine"

exec "$@"
