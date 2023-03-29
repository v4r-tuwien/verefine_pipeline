# Verefine Pipeline
Pipeline for detecting objects and estimating and refining their pose. 

The pipeline is implemented to use YCB-V objects and configured for the use on our Toyota HSR Sasha.

## Startup using the compose file(s)

The following commands will download the necessary data and then build all the docker containers and start them. 

If the containers were already built before, you can still use the same commands (except download_data.sh) to start the pipeline.

densefusion + verefine:
```
./download_data.sh
cd compose/densefusion_pipeline
xhost +
docker-compose up
```

Three Docker containers will be started:
- maskrcnn: [Mask-RCNN](https://github.com/matterport/Mask_RCNN) trained on YCB-V Dataset
- verefine: Pose Estimation with [DenseFusion](https://github.com/j96w/DenseFusion) and refinement using [VeREFINE](https://github.com/dornik/verefine)
- hsr-grasping: Node that calls detect, estimate_refine service and delivers object poses

ppf + verefine:
```
./download_data.sh
cd compose/densefusion_pipeline
xhost +
docker-compose up
```

Three Docker containers will be started:
- maskrcnn: [Mask-RCNN](https://github.com/matterport/Mask_RCNN) trained on YCB-V Dataset
- ppf: Pose Estimation with PPF and refinement using [VeREFINE](https://github.com/dornik/verefine)
- hsr-grasping: Node that calls detect, estimate_refine service and delivers object poses


## Build Dockerfile with global context

maskrcnn:
`docker build -t maskrcnn -f src/maskrcnn/Dockerfile .`

hsr-grasping:
`docker build -t hsr-grasping -f src/task/Dockerfile .`

verefine:
`docker build -t verefine -f src/task/Dockerfile .`

ppf:
`docker build -t verefine -f src/ppf/Dockerfile .`


## Visualization
In RVIZ you can view the final refined poses and the object segmentation by Mask-RCNN. 
They are published as images to these topics:
- ```/hsr_grasping/segmentation```
- ```/verefine/estimated_poses``` (densefusion)
- ```/ppf_estimation_and_verefine_refinement/verefine_debug``` (ppf)
 
## Service 
The pipeline will advertise a service ```/hsr_grasping/get_poses``` of the type [get_poses.srv](https://github.com/v4r-tuwien/object_detector_msgs/blob/main/srv/get_poses.srv). The response represents the refined poses of the detected objects in the camera frame.

The services that are internally called are 
- ```/hsr_grasping/detect_objects``` of the type [detectron2_service_server.srv](https://github.com/v4r-tuwien/object_detector_msgs/blob/main/srv/detectron2_service_server.srv) 
- ```/hsr_grasping/estimate_poses``` of the type [estimate_poses.srv](https://github.com/v4r-

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
### Config-Files
The params for camera, model paths and topics are in config/params.yaml
Change this according to your project

```
im_width  # input image widht
im_height: # input image height
intrinsics:
- [538.391033533567, 0.0, 315.3074696331638]  # camera intrinsics
- [0.0, 538.085452058436, 233.0483557773859]
- [0.0, 0.0, 1.0]  

cfg_dir: "/verefine/data"  # config directory to load ppf_pose_estimation_config.ini for ppf
models_dir: "/verefine/data/models/renderer"  # ppf model directory

ycbv_names: /verefine/data/ycbv_names.json  # ycbv mapping form *.ply to object name
ycbv_verefine: /verefine/data/ycbv_verefine.json  # necessary verefine information about the models

color_topic: /hsrb/head_rgbd_sensor/rgb/image_rect_color #  rgb image topic
depth_topic: /hsrb/head_rgbd_sensor/depth_registered/image_rect_raw  # depth image topic
camera_info_topic: /hsrb/head_rgbd_sensor/rgb/camera_info # camera info topic

debug_visualization: True # for ppf 
publish_tf: True
max_dist: 1.0  # crop points farther than this [meters]
down_scale: 1  # down_scale speeds things up by already subsampling the image [power of 2]

distance_threshold: 0.01 # for plane detection
to_meters: 0.001 # for plane detection
```

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
