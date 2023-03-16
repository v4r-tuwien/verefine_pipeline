# Verefine Pipeline
Pipeline for detecting objects and estimating and refining their pose. 

The pipeline is implemented to use YCB-V objects and configured for the use on our Toyota HSR Sasha.

## Startup using the compose file(s)

The following commands will download the necessary data and then build all the docker containers and start them. 

If the containers were already built before, you can still use the same commands (except download_data.sh) to start the pipeline. 

```
./download_data.sh
cd compose/densefusion_pipeline
xhost +local:'hostname'
docker-compose up
```
Three Docker containers will be started:
- detect: [Mask-RCNN](https://github.com/matterport/Mask_RCNN) trained on YCB-V Dataset
- estimate_refine: Pose Estimation with [DenseFusion](https://github.com/j96w/DenseFusion) and refinement using [VeREFINE](https://github.com/dornik/verefine)
- grasp: Node that calls detect, estimate and refine services and delivers object poses
 
## Visualization
In RVIZ you can view the final refined poses and the object segmentation by Mask-RCNN. 
They are published as images to these topics:
- ```/hsr_grasping/segmentation```
- ```/hsr_grasping/refined_poses```
 
## Service 
The pipeline will advertise a service ```/hsr_grasping/get_poses``` of the type [get_poses.srv](https://github.com/v4r-tuwien/object_detector_msgs/blob/main/srv/get_poses.srv). The response represents the refined poses of the detected objects in the camera frame.

The services that are internally called are 
- ```/hsr_grasping/detect_objects``` of the type [detectron2_service_server.srv](https://github.com/v4r-tuwien/object_detector_msgs/blob/main/srv/detectron2_service_server.srv) 
- ```/hsr_grasping/estimate_poses``` of the type [estimate_poses.srv](https://github.com/v4r-tuwien/object_detector_msgs/blob/main/srv/estimate_poses.srv) 
- ```/hsr_grasping/refine_poses``` of the type [refine_poses.srv](https://github.com/v4r-tuwien/object_detector_msgs/blob/main/srv/refine_poses.srv)

### Main Service
#### get_poses.srv
```
---
PoseWithConfidence[] poses
```
### Internal Services
#### detectron2_service_server.srv
```
sensor_msgs/Image image
---
object_detector_msgs/Detections detections
```

#### estimate_poses.srv
```
Detection det
sensor_msgs/Image rgb
sensor_msgs/Image depth
---
PoseWithConfidence[] poses
```

#### refine_poses.srv
```
Detection det
sensor_msgs/Image rgb
sensor_msgs/Image depth
PoseWithConfidence[] poses
---
PoseWithConfidence[] poses
```

### Important Messages
#### PoseWithConfidence.msg
```
string name
geometry_msgs/Pose pose
float32 confidence
```

#### Detecions.msg
```
Header header

uint32 height
uint32 width

Detection[] detections
```

#### Detection.msg
```
string name
float32 score
BoundingBox bbox
int64[] mask
```

## Configurations
### ROS Master
The ROS Master is set in the docker-compose.yml file for each container 
```
environment:
      ROS_MASTER_URI: "http://hsrb:11311"
```
### ROS Namespace
The Namespace is also defined in the docker-compose.yml file for each container. It is passed as command with the python script calls like this:
```
command: bash -c "source /maskrcnn/catkin_ws/devel/setup.bash; ROS_NAMESPACE=hsr_grasping python3 /maskrcnn/src/maskrcnn/ros_detection.py"
```

If you change it, the service names and visualization topics will change accordingly.

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
