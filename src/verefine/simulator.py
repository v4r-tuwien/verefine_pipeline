# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

import pybullet  # physics
import pybullet_data

import numpy as np
from scipy.sparse import csgraph
import logging
# logging.basicConfig(format='%(message)s', level=logging.INFO, filename='simulate.log')

import time
import os
import time

import util.rotation as r  # TODO is there some functionality like this in bop toolkit?


class Simulator:

    def __init__(self, dataset, instances_per_object=1, objects_to_use=[]):
        """
        :param dataset:
        :param instances_per_object: how many instances to deal with in parallel; != hypotheses to consider
        :param objects_to_use:
        """
        self.dataset = dataset
        self.instances_per_object = instances_per_object
        if len(objects_to_use) == 0:
            self.objects_to_use = dataset.objlist
        else:
            self.objects_to_use = objects_to_use  # allows to only use specific objects

        # --- Object handles
        self.planeId = None  # base plane
        self.models = dict()  # obj+collider loaded into pybullet -> handle in pybullet
        self.pyb = pybullet

        self.initialize_bullet()
        self.initialize_dataset()

        # --- Imported frame infos
        self.T_cv = None
        self.T_gl = None

        self.fixed_hypotheses = []

        self.runtimes = []

    def deinitialize(self):
        # TODO anything else to clean-up?
        pybullet.disconnect()

    def initialize_bullet(self):
        """
        Initialize physics world in headless mode or with GUI (debug).
        """
        # --- Bullet
        GUI = False
        if GUI:
            self.world = pybullet.connect(pybullet.GUI)
            pybullet.setRealTimeSimulation(0)
        else:
            self.world = pybullet.connect(pybullet.DIRECT)  # non-graphical client

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = pybullet.loadURDF("plane.urdf", globalScaling=1/10.0)  # scale down from 30x30m
        pybullet.setCollisionFilterGroupMask(self.planeId, -1, 1, 1)

        pybullet.setGravity(0, 0, -9.81)

    def initialize_dataset(self):
        """
        Load colliders for given dataset and initialize them in physics world.
        """
        # --- get models
        # visual_shapes = list()
        collision_shapes = list()

        i_models = []
        i_model = 1
        for obj_name, model_path in zip(self.dataset.obj_names.values(), self.dataset.model_paths):
            logging.info("   %s" % model_path)
            try:
                if i_model in self.dataset.objlist[1:]:
                    # TODO debug
                    if ("simple" in model_path
                            and os.path.exists(model_path.replace("textured_simple", "collider_simple"))):  # only on YCB
                        print("using concave collider %s" % (model_path.replace("textured_simple", "collider_simple")))
                        collider_path = model_path.replace("textured_simple", "collider_simple")
                    else:
                        print("let pybullet compute collider")
                        collider_path = model_path

                    # use adapted sizes if available
                    meshScale = self.dataset.obj_scales[i_model - 1] if hasattr(self.dataset, 'obj_scales')\
                        else self.dataset.mesh_scale
                    collision_shape = pybullet.createCollisionShape(shapeType=pybullet.GEOM_MESH,
                                                                    fileName=collider_path,
                                                                    meshScale=meshScale)
                    base_mass = self.dataset.obj_masses[i_model - 1]

                    model = pybullet.createMultiBody(baseMass=base_mass,
                                                     # [kg]; 0... static -> only add mass when active
                                                     baseVisualShapeIndex=-1,
                                                     baseCollisionShapeIndex=collision_shape,
                                                     basePosition=[i_model * 0.5, 0, 0],
                                                     baseOrientation=[1, 0, 0, 0],
                                                     baseInertialFramePosition=[0, 0, self.dataset.obj_coms[i_model - 1]])
                    collision_shapes.append(collision_shape)

                    print("%s (%i)" % (obj_name, i_model))
                    obj_str = "%0.2d" % i_model  # TODO this is overwritten -- unify among datasets
                    self.models[obj_str] = model
                    i_models.append(i_model)
                else:
                    print("no model with id %i" % i_model)
                #     model = pybullet.loadURDF("plane.urdf")
                #
                #     visual_shapes.append(None)
                #     collision_shapes.append(None)
                i_model += 1

            except Exception as ex:
                logging.error("Failed to load model at '%s'." % model_path)
                logging.error(ex.args[0])  # original error message
                self.models.clear()

        for hi in range(1, self.instances_per_object):
            for i_model, collision_shape in zip(i_models, collision_shapes):
                base_mass = self.dataset.obj_masses[i_model - 1]

                model = pybullet.createMultiBody(baseMass=base_mass,
                                                 # [kg]; 0... static -> only add mass when active
                                                 baseVisualShapeIndex=-1,
                                                 baseCollisionShapeIndex=collision_shape,
                                                 basePosition=[i_model * 0.5, 0, 0],
                                                 baseOrientation=[1, 0, 0, 0],
                                                 baseInertialFramePosition=[0, 0, self.dataset.obj_coms[i_model - 1]])

                self.models["%0.2d(%0.2d)" % (i_model, hi)] = model

    def initialize_frame(self, extrinsics, topview=False, focus=None):
        R, t = extrinsics[:3, :3], extrinsics[:3, 3]

        self.T_cv = np.matrix(np.eye(4))
        self.T_cv[0:3, 0:3] = R
        self.T_cv[0:3, 3] = t

        self.T_gl = np.matrix(np.eye(4))
        self.T_gl[0:3, 0:3] = R.T
        self.T_gl[0:3, 3] = -R.T * t

        # TODO these actually belong into the renderer imo
        if not topview:  # from real camera position
            eye = self.T_gl[0:3, 3].T
            forward = self.T_gl[0:3, 0:3] * np.matrix([0, 0, 1]).T
            up = self.T_gl[0:3, 0:3] * np.matrix([0, -1, 0]).T
            target = (eye + forward.T)
            self.vMatrix = pybullet.computeViewMatrix(eye.tolist()[0], target.tolist()[0], up.T.tolist()[0], self.world)
        else:
            target = focus
            eye = focus.copy()
            eye[2] += 1
            up = self.T_gl[:3, 2]/np.linalg.norm(self.T_gl[:3, 2])#[0, 1, 0]  # TODO could also align this to camera z
            self.vMatrix = pybullet.computeViewMatrix(eye, target, up.T.tolist()[0], self.world)

    def reset_objects(self, fixed_hypotheses=[]):
        for obj_str in self.models.keys():
            # set collision
            if obj_str in self.objects_to_use or obj_str in fixed_hypotheses:
                pybullet.setCollisionFilterGroupMask(self.models[obj_str], -1, 1, 1)
            else:
                pybullet.resetBasePositionAndOrientation(self.models[obj_str], [-100, -100, -100],
                                                         r.euler_to_quaternion([0] * 3),
                                                         self.world)
                pybullet.setCollisionFilterGroupMask(self.models[obj_str], -1, 0, 0)

    def fix_hypothesis(self, hypothesis):
        obj_str = hypothesis.id
        # 0... fix, >0... dynamic
        pybullet.changeDynamics(self.models[obj_str], -1, mass=0)
        self.fixed_hypotheses.append(obj_str)
    def unfix(self):
        for obj_str in self.fixed_hypotheses:
            mass = self.dataset.obj_masses[int(obj_str[:2])-1]
            pybullet.changeDynamics(self.models[obj_str], -1, mass=mass)
            self.fixed_hypotheses.remove(obj_str)

    def cam_to_world(self, in_cam):
        return self.T_gl * in_cam

    def world_to_cam(self, in_world):
        return self.T_cv * in_world

    def world_to_bullet(self, in_world, id):
        R = in_world[0:3, 0:3]
        orientation = r.matrix_to_quaternion(R)  # quaternion of form [w, x, y, z]
        orientation = orientation[1:] + [orientation[0]]  # pybullet expects [x, y, z, w]

        # position is offset by center of mass (origin of collider in physics world)
        position = in_world[0:3, 3] + (R * np.array([[0], [0], [self.dataset.obj_coms[int(id[:2]) - 1]]]))

        return position, orientation

    def bullet_to_world(self, position, orientation, id):
        in_world = np.matrix(np.eye(4))
        orientation = [orientation[-1]] + orientation[:-1]  # pybullet returns [x, y, z, w]
        in_world[0:3, 0:3] = r.quaternion_to_matrix(orientation)
        # position was offset by center of mass (origin of collider in physics world)
        in_world[0:3, 3] = np.matrix(position).T \
                    - (in_world[0:3, 0:3] * np.array([[0], [0], [self.dataset.obj_coms[int(id[:2]) - 1]]]))
        return in_world

    def cam_to_bullet(self, in_cam, id):
        return self.world_to_bullet(self.cam_to_world(in_cam), id)

    def bullet_to_cam(self, position, orientation, id):
        return self.world_to_cam(self.bullet_to_world(position, orientation, id))

    def initialize_solution(self, hypotheses):#, render_all_hypotheses=False):
        # --- prepare scene models
        trafos = dict()
        # if not render_all_hypotheses:
        for instance_hypotheses in hypotheses:
            for hypothesis in instance_hypotheses:
                obj_str = hypothesis.id
                if obj_str not in self.objects_to_use:
                    continue
                T_obj = hypothesis.transformation.copy()

                trafos[obj_str] = T_obj.copy()

                # --- to pybullet
                position, orientation = self.cam_to_bullet(T_obj, obj_str)

                pybullet.resetBasePositionAndOrientation(self.models[obj_str], position, orientation, self.world)  # also sets v to 0
        # else:
        #     for hypotheses_per_object in hypotheses:
        #         for hi, hypothesis in enumerate(hypotheses_per_object):
        #             obj_str = hypothesis.id
        #             if obj_str not in self.objects_to_use:
        #                 continue
        #             # obj_str += "" if hi == 0 else "(%0.2d)" % hi
        #             T_obj = hypothesis.transformation.copy()
        #
        #             trafos[obj_str] = T_obj.copy()
        #
        #             # --- to pybullet
        #             position, orientation = self.cam_to_bullet(T_obj, obj_str)
        #
        #             pybullet.resetBasePositionAndOrientation(self.models[obj_str], position, orientation,
        #                                                      self.world)  # also sets v to 0
        return trafos

    def simulate_no_render(self, obj_str, delta, steps, solver_iterations=10, sub_steps=0, fix_others=True):
        pybullet.resetBasePositionAndOrientation(self.planeId, [0,0,0], [0,0,0,1], self.world)
        pybullet.setCollisionFilterGroupMask(self.planeId, -1, 1, 1)

        # set-up simulation
        pybullet.setTimeStep(delta)
        pybullet.setPhysicsEngineParameter(fixedTimeStep=delta,
                                           numSolverIterations=solver_iterations, numSubSteps=sub_steps)

        if fix_others:
            for other_str in self.models.keys():
                if other_str not in self.objects_to_use:
                    continue
                if other_str != obj_str:
                    pybullet.changeDynamics(self.models[other_str], -1, mass=0)
                else:
                    pybullet.changeDynamics(self.models[other_str], -1, mass=self.dataset.obj_masses[int(obj_str[:2])-1])

        t_start = time.time()


        for i in range(steps):
            pybullet.stepSimulation()


        self.runtimes.append((time.time() - t_start))
        # print("physics took %ims" % (self.runtimes[-1] * 1000))

        # read-back transformation after simulation
        position, orientation = pybullet.getBasePositionAndOrientation(self.models[obj_str], self.world)
        position = list(position)
        orientation = list(orientation)

        T_phys = self.bullet_to_cam(position, orientation, obj_str)

        return T_phys

        # obj_ids = [self.dataset.objlist.index(int(obj_id[:2])) for obj_id in trafos.keys()]
        # Rs = [T[:3, :3] for T in trafos.values()]
        # ts = [T[:3, 3] for T in trafos.values()]
        # view_matrix = np.matrix(self.vMatrix).reshape(4, 4).T * self.T_gl
        # M = self.renderer.render(obj_ids, self.K, Rs, ts, mode=render_mode, view=view_matrix, proj=proj)
        # if render_mode == 'depth':
        #     M = np.zeros((480, 640, 3)), M, np.zeros((480, 640, 3))
        #     # M = None, M, None
        # elif render_mode == 'depth+seg':
        #     M = (np.zeros((480, 640, 3)),) + M
        #     # M = (None, ) + M
