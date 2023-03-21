# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

"""
PoseEstimation.srv -- using DenseFusion
---------------------------------------
in:
    Detection det
    sensor_msgs/Image rgb
    sensor_msgs/Image depth
out:
    PoseWithConfidence[] poses
"""

import numpy as np
import cv2 as cv
from PIL import Image as Img
from PIL import ImageDraw, ImageFont
from matplotlib import cm
import json
from scipy.spatial.transform.rotation import Rotation

# original VeREFINE
from verefine.simulator import Simulator
from verefine.verefine import Verefine
import verefine.config as config

# ROS wrapper
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Point, Quaternion
from object_detector_msgs.msg import BoundingBox, Detection, PoseWithConfidence
from object_detector_msgs.srv import estimate_poses, estimate_posesResponse
from object_detector_msgs.srv import refine_poses, refine_posesResponse
import ros_numpy

from util.renderer import EglRenderer as Renderer
from util.plane_detector import PlaneDetector
from util.dataset import YcbvDataset
from densefusion.densefusion import DenseFusion  # includes estimator and refiner


CAMERA_INFO = rospy.get_param('/pose_estimator/CAMERA_INFO')
ESTIMATE_MODE = 3

if __name__ == "__main__":

    publisher = rospy.Publisher("/verefine/estimated_poses", Image, queue_size=1)
    # VeREFINE utilities
    dataset = YcbvDataset()
    width = rospy.get_param('/pose_estimator/im_width')
    height = rospy.get_param('/pose_estimator/im_height')
    intrinsics = np.asarray(rospy.get_param('/pose_estimator/intrinsics'))
    ycbv_names_json = rospy.get_param('/pose_estimator/ycbv_names')
    simulator, renderer = Simulator(dataset), Renderer(dataset, width, height)
    plane_detector = PlaneDetector(width, height, intrinsics, down_scale=1)
    
    # DenseFusion
    df = DenseFusion(width, height)
    # wrap with VeREFINE (refinement and verification)
    vf = Verefine(dataset, df, simulator, renderer)

    def estimate(req):
        # parse input to numpy array
        rgb = req.rgb
        depth = req.depth
        header = rgb.header
        assert rgb.width == width and rgb.height == height
        rgb = ros_numpy.numpify(rgb)
        depth = ros_numpy.numpify(depth) / 1000

        # --- run VeREFINE
        # estimate initial poses using DenseFusion
        hypotheses = []
        observation_mask = np.zeros((height * width), dtype=np.uint8)
        
        detection = req.det
        
        # get bbox and mask
        roi = [detection.bbox.ymin, detection.bbox.xmin, detection.bbox.ymax, detection.bbox.xmax]
        # mask indices to image-sized binary mask
        det_mask = np.array(detection.mask)
        observation_mask = np.zeros((height * width), dtype=np.uint8)
        observation_mask[det_mask] = 1 # compose masks over all detections
        observation_mask = observation_mask.reshape((height, width))  

        # find id to name
        ycbv_names = None
        try:
            f = open(ycbv_names_json)
            ycbv_names = json.load(f)
            f.close()
        except:
            print("YCBV_names not found")
            
        det_id = -1
        for number in ycbv_names:
            if ycbv_names[number] == detection.name:
                det_id = int(number)

        # estimate initial pose with DenseFusion
        object_hypotheses = df.estimate(rgb, depth, intrinsics, roi, observation_mask, det_id,
                                            config.HYPOTHESES_PER_OBJECT)
        # order may change during verification - keep track of which pose belongs to which detection
        for hypothesis in object_hypotheses:
            hypothesis['detection_idx'] = det_id
        hypotheses.append(object_hypotheses)

        # prepare observation: determine supporting plane using MSAC, compute normals
        plane = plane_detector.detect(depth, observation_mask)
        normals = dataset.get_normal_image(depth)
        depth[observation_mask == 0] = 0
        normals[observation_mask == 0] = 0

        observation = {
            'depth': depth,
            'normal': normals,
            'dependencies': None,  # let VeREFINE compute dependencies
            'extrinsics': plane,
            'intrinsics': intrinsics
        }

        # refine/verify poses using VeREFINE
        # IN list of lists: num_objects x num_hypotheses_per -- OUT list: num_objects x 1 (best)
        mode = config.MODE
        mode = ESTIMATE_MODE
        vf.mode = np.clip(mode, 0, 5)
        refined_hypotheses = vf.refine(observation, hypotheses)

        # visualize best hypothesis per object
        vis = rgb.copy()
        colors = cm.get_cmap('tab10')
        for i, hypothesis in enumerate(refined_hypotheses):
            print([int(hypothesis['obj_id']) - 1])
            print([hypothesis['pose']])
            est_depth, _ = renderer.render([int(hypothesis['obj_id']) - 1], [hypothesis['pose']],
                                           observation['extrinsics'], observation['intrinsics'])
            contour, _ = cv.findContours(np.uint8(est_depth > 0), cv.RETR_CCOMP,
                                         cv.CHAIN_APPROX_TC89_L1)
            color = tuple([int(c * 255) for c in colors(hypothesis['detection_idx'])])
            vis = cv.drawContours(vis, contour, -1, color, 2, lineType=cv.LINE_AA)
        vis = Img.fromarray(vis)
        draw_detections = ImageDraw.Draw(vis)
        font = ImageFont.load_default()
        for i, hypothesis in enumerate(refined_hypotheses):
            detection = req.det
            color = tuple([int(c * 255) for c in colors(hypothesis['detection_idx'])])
            roi = [detection.bbox.ymin, detection.bbox.xmin, detection.bbox.ymax, detection.bbox.xmax]
            text = "{0} ({1:.2f})".format(detection.name, hypothesis['confidence'])
            text_w, text_h = font.getsize(text)
            #draw_detections.rectangle((roi[1], roi[0], roi[3], roi[2]))
            draw_detections.rectangle((roi[1], roi[2], roi[1] + text_w, roi[2] + text_h), fill=color)
            draw_detections.text((roi[1], roi[2]), text, font=font, fill='black')
        publisher.publish(ros_numpy.msgify(Image, np.asarray(vis), encoding="rgb8"))

        # to response format
        estimates = []
        instance_counts = {}
        for hypothesis in refined_hypotheses:
            estimate = PoseWithConfidence()
            #estimate.header = header
            # --- meta
            #if not detection.id in instance_counts:
            #    instance_counts[detection.id] = 0
            #estimate.instance = instance_counts[detection.id]
            #instance_counts[detection.id] += 1
            #estimate.id = detection.id
            estimate.name = detection.name
            estimate.confidence = hypothesis['confidence']
            print("{0} + {1}".format(estimate.name, estimate.confidence))
            # --- pose
            # # optional: to BOP coordinates
            # offset = np.eye(4)
            # offset[:3, 3] = dataset.obj_model_offset[int(hypothesis['obj_id']) - 1]
            # hypothesis['pose'] = hypothesis['pose'] @ offset
            estimate.pose = Pose()
            estimate.pose.position = ros_numpy.msgify(Point, hypothesis['pose'][:3, 3])
            estimate.pose.orientation = ros_numpy.msgify(Quaternion, Rotation.from_matrix(hypothesis['pose'][:3, :3]).as_quat())
            estimates.append(estimate)

        response = estimate_posesResponse()
        response.poses = estimates
        return response


    rospy.init_node("verefine_estimation")
    s = rospy.Service("estimate_poses", estimate_poses, estimate)
    print("Pose estimation with VeREFINE ready.")

    rospy.spin()
