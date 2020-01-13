# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

# import numpy as np
# import os
import glob

# from bop_toolkit_lib import dataset_params
# from bop_toolkit_lib import inout
#
# import socket


class YcbvDataset:
    """

    Load dataset given in BOP19 format (https://bop.felk.cvut.cz/datasets/).
    Based on BOP toolkit (https://github.com/thodan/bop_toolkit).

    """

    def __init__(self, base_path="/verefine/data/"):
        # specify dataset
        # self.dataset = 'ycbv'
        # self.dataset_split = 'test'
        # dataset_split_type = None
        # datasets_path = r'/verefine/data/bop19_local'
        #
        # self.dp_split = dataset_params.get_split_params(
        #     datasets_path, self.dataset, self.dataset_split, dataset_split_type)
        # self.img_width, self.img_height = self.dp_split['im_size']
        #
        # model_type = 'eval'  # None = default.
        # self.dp_model = dataset_params.get_model_params(
        #     datasets_path, self.dataset, model_type)

        # derived properties
        self.num_objects = 21
        self.objlist = list(range(22))

        # TODO load model_info.json for the dataset
        self.obj_masses = [  # TODO
            1,
            0.5,
            0.5,
            1,
            1,
            1,
            0.2,
            0.2,
            1,
            0.5,
            2,
            1,
            2,
            0.5,
            10,
            10,
            0.5,
            0.1,
            0.5,
            0.5,
            0.5
        ]
        self.obj_masses = [float(mass) for mass in self.obj_masses]
        self.obj_names = {
            -1: "000_ground",
            0: "002_master_chef_can",
            1: "003_cracker_box",
            2: "004_sugar_box",
            3: "005_tomato_soup_can",
            4: "006_mustard_bottle",
            5: "007_tuna_fish_can",
            6: "008_pudding_box",
            7: "009_gelatin_box",
            8: "010_potted_meat_can",
            9: "011_banana",
            10: "019_pitcher_base",
            11: "021_bleach_cleanser",
            12: "024_bowl",
            13: "025_mug",
            14: "035_power_drill",
            15: "036_wood_block",
            16: "037_scissors",
            17: "040_large_marker",
            18: "051_large_clamp",
            19: "052_extra_large_clamp",
            20: "061_foam_brick"
        }
        self.obj_coms = [
            -0.074,  # 0.067
            0.0,  # -0.112, # 0.102  # TODO would make it upright
            0.0,  # -0.092, #0.084  # TODO would make it upright
            -0.059,  # 0.043
            -0.084,  # 0.108
            -0.021,  # 0.013
            0.0,  # -0.035, #0.05  # TODO bbox is rotated; would make it lie down
            0.0,  # -0.035,  # TODO bbox is rotated; would make it lie down
            -0.050,  # 0.034
            -0.017,  # 0.020
            -0.147,  # 0.096
            -0.112,  # 0.139
            -0.025,  # 0.031
            -0.039,  # 0.042
            0.0,  # -0.115, #0.072  # TODO would make it lying all the time
            -0.120,  # 0.086
            -0.008,  # 0.008
            0.0,  # -0.010, #0.010  # TODO makes it unstable
            -0.019,  # 0.020
            -0.018,  # 0.018
            -0.017  # 0.034
        ]
        self.obj_coms = [com * 0.5 for com in self.obj_coms]

        self.base_path = base_path  # TODO dependent on BOP, not YCBV
        self.model_paths = sorted(glob.glob(
            self.base_path + "/models/*/textured_simple.obj"))  # -> obj_01 will have id 1 in segmentation mask
        self.mesh_scale = [1.0] * 3
        self.obj_scales = [
            [1.00, 1.00, 0.94],  # 0: "002_master_chef_can",
            [1.00, 0.96, 1.00],  # 1: "003_cracker_box",
            [0.85, 1.00, 0.96],  # 2: "004_sugar_box",
            [1.00, 1.00, 0.95],  # 3: "005_tomato_soup_can",
            [1.00, 1.00, 0.97],  # 4: "006_mustard_bottle",
            [1.00, 1.00, 0.92],  # 5: "007_tuna_fish_can",
            [1.00, 0.94, 0.96],  # 6: "008_pudding_box",
            [0.95, 1.00, 0.97],  # 7: "009_gelatin_box",
            [0.92, 1.00, 1.00],  # 8: "010_potted_meat_can",
            [1.00, 1.00, 1.00],  # 9: "011_banana",
            [1.00, 1.00, 0.96],  # 10: "019_pitcher_base",
            [1.00, 1.00, 1.00],  # 11: "021_bleach_cleanser",
            [1.00, 1.00, 1.00],  # 12: "024_bowl",
            [1.00, 1.00, 0.90],  # 13: "025_mug",
            [1.00, 0.97, 0.93],  # 14: "035_power_drill",
            [1.00, 0.97, 0.90],  # 15: "036_wood_block",
            [1.00, 1.00, 1.00],  # 16: "037_scissors" TODO
            [1.00, 1.00, 1.00],  # 17: "040_large_marker",
            [1.00, 1.00, 1.00],  # 18: "051_large_clamp",
            [1.00, 1.00, 1.00],  # 19: "052_extra_large_clamp",
            [1.00, 1.00, 0.94],  # 20: "061_foam_brick"
        ]

# import open3d as o3d
import numpy as np


class ExApcDataset:
    """

    TODO

    """

    def __init__(self, base_path):
        self.num_objects = 11
        self.objlist = list(range(12))

        self.obj_masses = [1] * 11  # TODO
        self.obj_masses = [float(mass) for mass in self.obj_masses]
        self.obj_names = {
            -1: "table",
            0: "crayola_24_ct",
            1: "expo_dry_erase_board_eraser",
            2: "folgers_classic_roast_coffee",
            3: "scotch_duct_tape",
            4: "up_glucose_bottle",
            5: "laugh_out_loud_joke_book",
            6: "soft_white_lightbulb",
            7: "kleenex_tissue_box",
            8: "dove_beauty_bar",
            9: "elmers_washable_no_run_school_glue",
            10: "rawlings_baseball"
        }
        self.obj_coms = [0.0] * 11  # TODO
        self.obj_coms = [com * 0.5 for com in self.obj_coms]

        self.base_path = base_path
        obj_names = list(self.obj_names.values())[:-1]  # key -1 is last
        self.model_paths = [self.base_path + "models/%s/%s.obj" % (obj_name, obj_name) for obj_name in obj_names]
        self.mesh_scale = [1.0] * 3  # TODO
        self.obj_scales = [[1.0, 1.0, 1.0]] * 11  # TODO

        self.pcd = []
        for obj_name in list(self.obj_names.values())[:-1]:
            with open(self.base_path + "models/%s/points.xyz" % obj_name,
                      'r') as file:
                pts = file.readlines()

            cld = []
            for pt in pts:
                pt = pt.split(" ")
                pt = [float(v.replace("\n", "")) for v in pt]
                cld.append(pt)

            self.pcd.append(np.array(cld))

