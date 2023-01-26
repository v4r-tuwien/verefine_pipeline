#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/melodic/setup.bash"
source "/visualization/catkin_ws/devel/setup.bash"

# start roscore tmux session -- user can use the second pane to call /visualize_ycbv
tmux new-session -d -s "visualization"
tmux splitw -h -p 50
tmux selectp -t 0
tmux splitw -v -p 50
tmux selectp -t 0
tmux send-keys "roscore" C-m
tmux selectp -t 1
tmux send-keys "ROS_NAMESPACE=verefine python3 /visualization/src/ros_visualize_ycbv.py" C-m
tmux selectp -t 2
tmux splitw -v -p 50
tmux selectp -t 2
tmux send-keys "rviz -d ./src/visualize_ycbv.rviz" C-m
tmux selectp -t 3
tmux send-keys "rosservice call /verefine/visualize_ycbv 59 1070 5"  # user has to hit enter when perception is started

tmux attach-session -t "visualization"

exec "$@"
