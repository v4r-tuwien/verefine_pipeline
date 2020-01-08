# hsr-grasping

## Build Dockerfile with global context
In hsr-grasping:

`docker build -f src/densefusion/Dockerfile . -t densefusion`

## Run densefusion
`docker run -it --runtime=nvidia --net=host --volume=/home/dominik/projects/hsr-grasping:/hsr-grasping densefusion`

to prevent ros_pose.sh from executing, add --entrypoint="/bin/bash"

## Run verefine
`xhost +local:'hostname'` \
`docker run -it --net=host --env='DISPLAY' --env='QT_X11_NO_MITSHM=1' --volume=/home/dominik/projects/hsr-grasping:/hsr-grasping verefine /bin/bash`

## Run ppf
TODO mount model dir
in container...
cd to home\
`rosrun ppf_recognizer_ros recognition_service _models_dir:=/path/to/models _cfg_dir:=/path/to/config`
