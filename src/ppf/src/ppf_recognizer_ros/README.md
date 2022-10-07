# ROS Wrapper for ppf recognizer extracted from the v4r library

Start the recognizer using :

```bash
rosrun ppf_recognizer_ros recognition_service _models_dir:=/home/v4r/models _cfg_dir:=/home/v4r/catkin_ws/src/ppf_recognizer_ros/cfg
```

or :

```bash
roslaunch ppf_recognizer_ros recognition_service.launch models:="/path/to/ppf/models" config:="/path/to/config.ini"
```

If the models or config args are missing the default values : "/home/v4r/models", "/home/v4r/catkin_ws/src/ppf_recognizer_ros/cfg/ppf_pose_estimation_config.ini" are assumed.

Additional arguments can be passed with args:=" -arg1 [-arg2 [...]]".
For example:

- args:="-h" to see all available arguments.
- args:="--models \"021_bleach_cleanser 002_master_chef_can\" " to only load those two models instead of all of them.
