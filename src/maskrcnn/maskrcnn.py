import numpy as np
import tensorflow as tf_backend
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from mrcnn import utils
import mrcnn.model as modellib

from src.maskrcnn.maskrcnn_util import BopInferenceConfig


class MaskRcnnDetector:

    """
    Mask R-CNN detector
    """

    def __init__(self):
        cfg = tf_backend.ConfigProto()
        cfg.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.graph = tf_backend.Graph()
        self.session = tf_backend.Session(config=cfg)
        tf_backend.keras.backend.set_session(self.session)

        n_objs = 21  # TODO load

        '''
        standard estimation parameter for Mask R-CNN (identical for all dataset)
        '''
        with self.graph.as_default():
            self.config = BopInferenceConfig(dataset="ros",
                                             num_classes=n_objs + 1,
                                             im_width=640, im_height=480)  # TODO load
            self.config.DETECTION_MIN_CONFIDENCE = 0.3
            self.config.DETECTION_MAX_INSTANCES = 30
            self.config.DETECTION_NMS_THRESHOLD = 0.5

            self.detection_model = modellib.MaskRCNN(mode="inference", config=self.config, model_dir="/")
            self.detection_model.load_weights("/maskrcnn/data/maskrcnn_ycbv.h5", by_name=True)  # TODO load

            self.detection_labels = list(range(1, 22))  # TODO load

    def detect(self, rgb):
        """
        TODO taken from Pix2Pose -- adapt
        :param observation:
        :return:
        """
        with self.graph.as_default():
            image_t = rgb

            image_t_resized, window, scale, padding, crop = utils.resize_image(
                np.copy(image_t),
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            if (scale != 1):
                print("Warning.. have to adjust the scale")
            results = self.detection_model.detect([image_t_resized], verbose=0)
            r = results[0]
            rois = r['rois']
            rois = rois - [window[0], window[1], window[0], window[1]]

            obj_orders = np.array(r['class_ids']) - 1
            obj_ids = []
            for obj_order in obj_orders:
                obj_ids.append(obj_order+1)

            # now c_ids are the same annotation those of the names of ply/gt files
            scores = np.array(r['scores'])
            masks = r['masks'][window[0]:window[2], window[1]:window[3], :]

            return obj_ids, rois, masks, scores
