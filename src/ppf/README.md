Implements the LocateObject action

## DESCRIPTION:
- segmentation: plane pop-out
- object pose estimation: point pair features (PPF)
- refinement (optional): use Point-to-Plane ICP to refine all initial hypotheses
- verification (optional): rendering-based scoring of PPF hypotheses, selects best per object
- VeREFINE (optional): physics-guided multi-object/multi-hypotheses refinement and verification

## PREPARATION:
- The current version of this package is designed for spray-painted canisters (we created ours using [this spray](https://www.amazon.de/gp/product/B000R9QHWY/ref=ppx_yo_dt_b_asin_title_o01_s00?ie=UTF8&psc=1), any colour should work, as long as it is matte enough to get good depth data)
- unpack [`data.zip`](https://owncloud.tuwien.ac.at/index.php/s/hsKFDtalkilCo83) (download pw: _tracebot_) to get models and examples
- if you already used a previous version of this package, run `docker-compose build locate_object` to make sure you are using the latest image

## USAGE:

### Fully containerized
- Depending on the permissions you set up, you might need to grant access to the X server to docker with `xhost local:docker`
- Run `docker-compose run locate_object` - this builds image on first run (see `docker-compose.yml`)
- Start the camera
  - _off-line_: Use `ros_dummy_camera.py` to automatically publish example camera images and runs the action server using `ros_dummy_client.py` (see the Test the installation section)
  - _on-line_: start the D435 using `roslaunch realsense2_camera rs_aligned_depth.launch`
- note: optionally, PPF will generate hashes for the new models on the first run (see `parse_ply.py`); if you use the data provided above, the pre-computed hashes for container and tray are already included

### With a system-wide ROS install
- Start the camera using `roslaunch realsense2_camera rs_aligned_depth.launch` and run `roslaunch locateobject locate_object.launch`
- Alternatively `roslaunch locateobject locate_object_w_camera.launch` to launch both the camera and the node at once

### Test the installation
- Start the action server with your preferred method
- Make sure you have downloaded the data archive (see above)
- Start the dummy camera with

```
docker exec -it `docker ps -q --filter "name=locateobject_locate_object*"` bash -c '/ros_entrypoint.sh python3 src/locate_object/ros_dummy_camera.py'
```

- You can trigger the action server with the following command 
```
docker exec -it `docker ps -q --filter "name=locateobject_locate_object*"` bash -c '/ros_entrypoint.sh python3 src/locate_object/ros_dummy_client.py'
```
