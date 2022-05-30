# hsr-grasping

## Using the compose file(s)

The following commands will download the necessary data and then build all the docker containers and start them. 

If the containers were already built before, you can still use the same commands (except download_data.sh) to start the pipeline. 

```
./download_data.sh
cd compose/densefusion_pipeline
xhost +local:'hostname'
docker-compose up
```

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

download the ppf models from https://drive.google.com/open?id=1CUICC6UGZI8e4AXeu48T0w9uj7UX7WWq

extract to data/ycbv_ppf_models

run the container with:

`docker run -it --net=host --volume=/path/to/data/ycbv_ppf_models:/home/v4r/models`

in the container run :

`rosrun ppf_recognizer_ros recognition_service _models_dir:=/path/to/models _cfg_dir:=/path/to/config`

or :

```bash
roslaunch ppf_recognizer_ros recognition_service.launch models:="/path/to/ppf/models" config:="/path/to/config.ini"
```

If the models or config args are missing the default values : "/home/v4r/models", "/home/v4r/catkin_ws/src/ppf_recognizer_ros/cfg/ppf_pose_estimation_config.ini" are assumed.

Additional arguments can be passed with args:=" -arg1 [-arg2 [...]]".
For example:

- args:="-h" to see all available arguments.
- args:="--models \"021_bleach_cleanser 002_master_chef_can\" " to only load those two models instead of all of them.

