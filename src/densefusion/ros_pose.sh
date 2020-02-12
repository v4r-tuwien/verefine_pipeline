#!/bin/bash

tmux new-session -d -s densefusion
tmux set -g mouse on

tmux new-window -a  densefusion

tmux split-window -h
tmux select-pane -t 0
tmux send-keys "export ROS_MASTER_URI=http://10.0.0.143:11311" C-m
tmux send-keys "source /densefusion/catkin_ws/devel/setup.bash" C-m
tmux select-pane -t 1
tmux send-keys "export ROS_MASTER_URI=http://10.0.0.143:11311" C-m
tmux send-keys "source /densefusion/catkin_ws/devel/setup.bash" C-m
tmux select-pane -t 0
tmux send-keys "python src/densefusion/ros_pose_estimation.py" C-m
tmux send-keys enter
tmux select-pane -t 1
tmux send-keys "python src/densefusion/ros_pose_refinement.py" C-m
tmux send-keys enter
tmux rename-window 'densefusion'

tmux attach-session -t densefusion

