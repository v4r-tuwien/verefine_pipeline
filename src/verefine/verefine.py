# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from drawnow import drawnow

import numpy as np
from scipy.spatial.transform.rotation import Rotation

# make reproducible (works up to BAB -- TODO VF smh not)
# seed = 0
# np.random.seed(seed)
# import random
# random.seed = 0

# ===================
# ===== CONFIG ======  TODO set via parameters in experiment
# ===================
USE_BASELINE_HYPOTHESES = False  # whether to use precomputed baseline hypotheses pool (True) or recompute (False)
USE_POSECNN = False  # load posecnn's results as pose estimator -- note: only one hypothesis per object
USE_COLLISION_CLUSTERING = True  # else rendering-based version
USE_ICP = False  # else use DF as refiner
FIX_OTHERS = True  # whether to fix supporting objects in their best hypothesis (True) or simulate them as well (False)
EXPAND_ADJACENT = False  # whether to only expand adjacent objects (True) or based on <2cm displacement (False)
TRACK_CONVERGENCE = False  # get best hypotheses at each iteration of SV (1 iter ~ [OV budget] iterations of plain MCTS)
# POLICY = UCB
C = 1  # TODO was 1e-1

ALL_OBJECTS = True
HYPOTHESES_PER_OBJECT = 25
ITERATIONS = 1
REFINEMENTS_PER_ITERATION = 1
SIM_STEPS = 60  # 60 as in Mitash, 3 as in paper (~10*5ms)

BUDGET_SCALE = 1 if not ALL_OBJECTS else 3  # scales max iterations of verification (1... nobject * max iter per object -> one wrong node and we cannot refine all)
REFINE_AT_ONCE = True  # during SV, spend budget one-by-one (False) or all at once (True)

MAX_REFINEMENTS_PER_HYPOTHESIS = ITERATIONS * REFINEMENTS_PER_ITERATION * HYPOTHESES_PER_OBJECT  # BUDGET
MAX_MCTS_ITERATIONS_PER_OBJECT = MAX_REFINEMENTS_PER_HYPOTHESIS + HYPOTHESES_PER_OBJECT  # BUDGET + num hypotheses (initial)
USE_HEURISTIC_EXPANSION = False
DEBUG_PLOT = False
DEBUG_PLOT_REFINE = False
TAU, TAU_VIS = 20, 10  # [mm]
ALPHA = np.cos(np.deg2rad(45))

# FIX_N_SKIP = False  # skip frames with bad plane estimates and fix z-offset for all others
REPETITIONS = 1
BUDGET = ITERATIONS * REFINEMENTS_PER_ITERATION * HYPOTHESES_PER_OBJECT

SIMULATOR = None
RENDERER = None
REFINER = None
OBSERVATION = None

# ==================
# ===== COMMON =====
# ==================
cost_durations = []

def fit(observation, rendered, unexplained):
    st = time.time()
    depth_obs = observation['depth'].copy()
    depth_ren = rendered[1] * 1000  # in mm TODO or do this in renderer?

    # mask = np.logical_and(depth_ren > 0, depth_obs > 0)# TODO or just np.logical_and(mask, depth_ren>0)#
    # mask = depth_ren>0
    mask = np.logical_or(depth_ren>0, depth_obs>0)  # TODO if depth_obs is already masked
    # mask = depth_obs > 0
    # mask = np.logical_and(mask, np.logical_not(observation["mask_others"]))
    # mask = np.logical_not(observation["mask_others"])
    if unexplained is not None:
        mask = np.logical_and(unexplained, mask)  #depth_ren>0#

    if np.count_nonzero(mask) == 0 or (depth_ren > 0).sum() == 0:  # no valid depth values
        cost_durations.append(time.time() - st)
        return 0

    # masks
    depth_vis_test = depth_obs.copy()
    depth_vis_test[depth_obs == 0] = 1000

    depth_mask = mask#np.logical_and(mask, depth_ren - depth_vis_test < TAU_VIS)  # only visible -- ren at most [TAU_VIS] behind obs
    cos_mask = depth_mask#np.logical_and(mask, depth_ren - depth_obs < 10)
    #
    # # visibility
    # visibility_ratio = float(depth_mask.sum()) / float((depth_ren > 0).sum())  # visible / rendered count
    # visibility_ratio = float((depth_mask > 0).sum()) / float((mask > 0).sum())  # visible / total

    # delta fit
    dist = np.abs(depth_obs[depth_mask] - depth_ren[depth_mask])
    delta = np.mean(np.min(np.vstack((dist / TAU, np.ones(dist.shape[0]))), axis=0))

    # # cos fit
    # norm_obs = observation['normals']
    # norm_ren = rendered[3]
    # # # plt.imshow((norm_ren+1)/2)
    # # # plt.show()
    # on = norm_obs[cos_mask].reshape(cos_mask.sum(), 3)
    # rn = norm_ren[cos_mask].reshape(cos_mask.sum(), 3)
    # cos = np.einsum('ij,ij->i', on, rn)
    # cos[cos < 0] = 0  # clamp to zero
    # # cos_fit = np.mean(cos)  # TODO or also linear until threshold?
    # # print("cos =%0.2f" % np.mean(cos))
    # # print("dist=%0.2f" % (1-delta))
    # #
    # # # # Aldoma
    # # assert depth_mask.shape[0] == cos_mask.shape[0]
    # aldoma_delta = (1 - dist/TAU) * cos
    # aldoma_delta[dist > TAU] = 0
    # fit = np.mean(aldoma_delta)

    # TODO mask already removes explained stuff -- so we need a way to have same size of delta and lamba w/o this
    # aldoma_lambda = np.logical_not(np.logical_and(unexplained == 0, depth_mask > 0))[mask]  # already explained and would be visible
    # fit = np.mean((aldoma_delta + aldoma_lambda)/2)
    # if (unexplained == 0).sum() == 0:
    #     lmbda = 1.0
    # else:
    #     lmbda = np.mean(np.abs(depth_obs - depth_ren)[unexplained == 0] > 15)

    # TODO clutter term
    # ...

    # visible = depth_ren - depth_obs < TAU_VIS
    # if (visible > 0).sum() < 1000:
    #     msk = np.logical_and(mask, visible)
    #     dst = np.abs(depth_obs[msk] - depth_ren[msk])
    #     delta = np.mean(np.min(np.vstack((dst / TAU, np.ones(dst.shape[0]))), axis=0))

    # TODO multi assignment + scaling per term
    # fit = visibility_ratio * (1 - delta) * cos_fit
    # fit = visibility_ratio * (1 - delta)
    # fit = ((1-delta) + lmbda)/2
    # fit = (1-delta)*0.95 + np.mean(cos)*0.05
    fit = 1-delta

    # Mitash score + normalization TODO check if [0,1]
    # depth_obs = depth_obs[mask > 0]
    # depth_ren = depth_ren[mask > 0]  # TODO renders state i+1 over rendering of state i
    # depth_others = observation["depth_others"]
    # depth_ren[depth_ren==0] = 1000
    # depth_ren[np.logical_and(depth_ren>depth_others, depth_others > 0)] = depth_others[np.logical_and(depth_ren>depth_others, depth_others > 0)]
    # depth_ren[depth_ren==1000] = 0
    # # if (depth_others > 0).sum() > 0:
    # #     plt.imshow(depth_ren)
    # #     plt.show()
    #
    # obScore = (np.logical_and(depth_obs > 0, np.abs(depth_obs - depth_ren) > 10) > 0).sum()
    # renScore = (np.logical_and(depth_ren > 0, np.abs(depth_obs - depth_ren) > 10) > 0).sum()
    # intScore = (np.logical_and(np.logical_and(depth_obs > 0, depth_ren > 0), np.abs(depth_obs - depth_ren) > 10) > 0).sum()
    # fit = 1 - ((obScore + renScore - intScore) / (np.logical_or(depth_obs > 0, depth_ren > 0) > 0).sum())


    if np.isnan(fit):  # invisible object
        cost_durations.append(time.time() - st)
        return 0

    cost_durations.append(time.time() - st)
    return fit

def fit_single(observation, rendered, unexplained):
    st = time.time()
    depth_obs = observation['depth'].copy()
    depth_ren = rendered[1] * 1000  # in mm TODO or do this in renderer?
    norm_obs = observation['normals']
    norm_ren = rendered[3]

    mask = np.logical_or(depth_ren>0, depth_obs>0)  # TODO if depth_obs is already masked
    # mask = np.logical_and(depth_ren>0, depth_obs>0)
    # if unexplained is not None:
    #     mask = np.logical_and(unexplained, mask)

    # if "label" in observation:
    #     mask = np.logical_and(mask, observation["label"] > 0)

    # depth_vis_test = depth_obs.copy()
    # depth_vis_test[depth_obs == 0] = 1000
    # vis_mask = np.logical_and(mask,
    #                           depth_ren - depth_vis_test < TAU_VIS)  # only visible -- ren at most [TAU_VIS] behind obs
    # visibility = float((vis_mask>0).sum()) / float((depth_ren>0).sum()) if np.count_nonzero(mask) > 0 else 0
    # mask = vis_mask

    if np.count_nonzero(mask) == 0 or (depth_ren > 0).sum() == 0:  # no valid depth values
        cost_durations.append(time.time() - st)
        return 0

    # vis_mask = depth_ren - depth_vis_test < 5#TAU_VIS
    # overlap = 1 - float(np.logical_and(depth_ren>0, observation['mask_others']).sum()) / float((depth_ren>0).sum())
    # overlap = 1 - float(np.logical_and(depth_ren>0, depth_obs>0).sum()) / float((depth_ren>0).sum())
    # overlap = 1 - float(np.logical_and(vis_mask, depth_obs > 0).sum()) / float(vis_mask.sum())

    # unsupported
    unsupported = float(np.logical_and(depth_ren > 0, depth_obs == 0).sum()) / float((depth_ren>0).sum())

    # TODO debug - perfect fit
    # depth_ren[depth_ren>0] = depth_obs[depth_ren>0]
    # norm_ren[depth_ren>0] = norm_obs[depth_ren>0]

    # delta fit
    dist = np.abs(depth_obs[mask] - depth_ren[mask])
    delta = np.mean(np.min(np.vstack((dist / TAU, np.ones(dist.shape[0]))), axis=0))

    # cos fit
    # norm_ren = np.dstack([norm_ren[:,:,0], norm_ren[:,:,2], norm_ren[:,:,1]])
    # # plt.imshow((norm_ren+1)/2)
    # # plt.show()
    on = norm_obs[mask].reshape(mask.sum(), 3)
    rn = norm_ren[mask].reshape(mask.sum(), 3)
    cos = np.einsum('ij,ij->i', on, rn)
    cos[cos < 0] = 0  # clamp to zero
    cos = 1 - np.min(np.vstack(((1-cos) / ALPHA, np.ones(cos.shape[0]))), axis=0)  # threshold and normalize
    cos_fit = np.mean(cos)

    # aldoma_delta = (1 - dist/TAU) * cos
    # aldoma_delta[dist > TAU] = 0
    # fit = np.mean(aldoma_delta)

    fit = (1-delta) * 0.5 + cos_fit * 0.5
    # print(fit)
    # fit = (1-delta)*0.4 + cos_fit*0.4 + (1-unsupported) * 0.2
    # fit = fit * (fit) + (1-unsupported)*(1-fit)
    # fit /= ((depth_ren > 0).sum() / (mask > 0).sum())
    # fit *= visibility

    # if "label" in observation:
    #     overlap = np.logical_and(depth_ren>0, observation["label"] > 0).sum() / (depth_ren>0).sum()
    #     fit = fit * 0.7 + overlap * 0.3

    if np.isnan(fit):  # invisible object
        cost_durations.append(time.time() - st)
        return 0

    cost_durations.append(time.time() - st)
    return fit


def fit_multi(observation, rendered, unexplained):
    st = time.time()
    depth_obs = observation['depth'].copy()
    depth_ren = rendered[1] * 1000  # in mm TODO or do this in renderer?

    # mask = np.logical_or(depth_ren > 0, depth_obs > 0)
    # mask = np.logical_and(mask, np.logical_not(observation["mask_others"]))
    mask = np.logical_not(observation["mask_others"])
    # if unexplained is not None:
    #     mask = np.logical_and(unexplained, mask)
    if np.count_nonzero(mask) == 0 or (depth_ren > 0).sum() == 0:  # no valid depth values
        cost_durations.append(time.time() - st)
        return 0

    # masks
    depth_vis_test = depth_obs.copy()
    depth_vis_test[depth_obs == 0] = 1000
    # n_all = mask.sum()
    vis_mask = depth_ren - depth_vis_test < TAU_VIS  # only visible -- ren at most [TAU_VIS] behind obs
    depth_mask = np.logical_and(mask, vis_mask)
    # n_vis = depth_mask.sum()
    # p_vis = float(n_vis) / float(n_all)

    # unsupported
    unsupported = float(np.logical_and(depth_ren > 0, depth_obs == 0).sum()) / float((depth_ren > 0).sum())

    # double assignment
    # vis_rendered = np.logical_and(depth_ren > 0, vis_mask)
    # double_assign = float(np.logical_and(vis_rendered > 0, observation["mask_others"] > 0).sum()) / float((vis_rendered > 0).sum())

    # delta fit
    dist = np.abs(depth_obs[depth_mask] - depth_ren[depth_mask])
    # dist[observation["mask_others"][depth_mask > 0] > 0] = TAU/2
    delta = np.mean(np.min(np.vstack((dist / TAU, np.ones(dist.shape[0]))), axis=0))

    # cos fit
    norm_obs = observation['normals']
    norm_ren = rendered[3]
    # # plt.imshow((norm_ren+1)/2)
    # # plt.show()
    on = norm_obs[depth_mask].reshape(depth_mask.sum(), 3)
    rn = norm_ren[depth_mask].reshape(depth_mask.sum(), 3)
    cos = np.einsum('ij,ij->i', on, rn)
    cos[cos < 0] = 0  # clamp to zero
    # cos[observation["mask_others"][depth_mask > 0] > 0] = ALPHA/2
    cos = 1 - np.min(np.vstack(((1 - cos) / ALPHA, np.ones(cos.shape[0]))), axis=0)  # threshold and normalize
    cos_fit = np.mean(cos)

    fit = (1-delta)*0.5 + cos_fit*0.5  # 0.9,0.1 for 3 obj
    # fit = (1-delta)*0.45 + cos_fit*0.45 + (1-double_assign) * 0.1
    # fit /= ((depth_ren>0).sum()/(depth_obs>0).sum())
    #
    # if "label" in observation:
    #     overlap = np.logical_and(depth_ren>0, observation["label"] > 0).sum() / (depth_ren>0).sum()
    #     fit = fit * 0.9 + overlap * 0.1

    if np.isnan(fit):# or p_vis < 0.1:  # invisible object
        cost_durations.append(time.time() - st)
        return 0

    cost_durations.append(time.time() - st)
    return fit

fit_fn = fit_single


class Hypothesis:
    """
    A hypothesis giving a candidate object (model) and a candidate transformation.
    - "We assume that the object is present in the scene, under the given transformation."
    """

    def __init__(self, model, transformation, roi, mask, embedding, cloud, instance_id=0, confidence=1.0,
                 refiner_param=None):
        self.model = model
        self.instance_id = instance_id
        self.transformation = transformation
        # self.object = model
        self.id = "%s(%0.2d)" % (self.model, instance_id) if instance_id > 0 else self.model
        self.confidence = confidence

        self.roi = roi  # region of interest (relative to full image)
        self.mask = mask  # segmentation + depth mask (on full image)
        self.embedding = embedding  # from DF
        self.cloud = cloud  # from DF

        self.refiner_param = refiner_param

    def duplicate(self):
        """
        TODO
        :return:
        """
        return Hypothesis(model=self.model, transformation=self.transformation.copy(), roi=self.roi, mask=self.mask,
                          embedding=None,  # TODO compute embedding ad-hoc without estimation (for loading)
                          cloud=None,  # TODO for reproducibility -- the samples that were selected
                          instance_id=self.instance_id, confidence=self.confidence, refiner_param=self.refiner_param)

    def render(self, observation, mode, fixed=[]):
        """
        TODO
        :param observation:
        :param mode: color, depth, depth+seg or color+depth+seg
        :return: color in TODO, depth in meters, instance segmentation TODO
        """
        obj_id = RENDERER.dataset.objlist.index(int(self.id[:2]))  # TODO do this conversion in renderer

        assert mode in ['color', 'depth', 'depth+seg', 'depth+normal', 'color+depth+seg', 'cost', 'cost_multi']

        # TODO just pass a list of hypotheses alternatively
        obj_ids = [obj_id] + [RENDERER.dataset.objlist.index(int(other[0].id[:2])) for other in fixed]
        transformations = [self.transformation] + [other[0].transformation for other in fixed]
        rendered = RENDERER.render(obj_ids, transformations,
                                   observation['extrinsics'], observation['intrinsics'],
                                   mode=mode, cost_id=None)#(1 << obj_id))  # if mode is cost, only compute cost for this hypothesis

        return rendered

    def fit(self, observation, unexplained=None):
        """
        TODO mention PoseRBPF
        :param observation:
        :param rendered:
        :return:
        """
        # #
        # if fit_fn == fit_multi:# or HYPOTHESES_PER_OBJECT == 25:  # TODO also transfer to GPU
        # # # if HYPOTHESES_PER_OBJECT == 25:
        # #     # a) render depth, compute score on CPU
        # rendered = self.render(observation, mode='depth+normal')
        # #     # rendered[0] = self.render(observation, mode='color')[0]
        # #     # unexplained = np.ones_like(self.mask)# TODO self.mask
        # #     # for hs in fixed:
        # #     #     h_mask = hs[0].render(observation, mode='depth+seg')[2]
        # #     #     unexplained[h_mask > 0] = 0
        # score = fit_fn(observation, rendered, unexplained)
        # # else:  # fit_fn == fit_single
        # _, score = self.render(observation, mode='cost')
        _, score = self.render(observation, mode=('cost' if fit_fn == fit_single else 'cost_multi'))
        # print(np.abs(score-gpu_score))

        # b) render depth, compute score on GPU  TODO has a bug when object is too far off
        # _, score = self.render(observation, "cost")

        return score


# ==================
# ===== PHYSIR =====
# ==================


class PhysIR:

    """
    TODO
    """

    def __init__(self, refiner, simulator):

        self.refiner = refiner
        self.simulator = simulator
        self.fixed = []

        # TODO where to move this config stuff?
        self.PHYSICS_STEPS = 20
        self.REFINEMENTS_PER_ITERATION = 1

    def refine(self, hypothesis, override_iterations=-1, explained=None):
        """

        :param hypothesis:
        :return:
        """

        # TODO check if this is correct
        if override_iterations == -1:
            iterations = hypothesis.refiner_param[-4]
            hypothesis.refiner_param[-4] = 1  # TODO or Refs_per_iter?
        else:
            iterations = override_iterations
            hypothesis.refiner_param[-4] = 1
        # iterations = hypothesis.refiner_param[-4] if override_iterations == -1 else override_iterations
        self.simulator.objects_to_use = [hypothesis.id] + [fix[0].id for fix in self.fixed]
        self.simulator.reset_objects()

        physics_hypotheses = []
        for ite in range(0, iterations):
            # === PHYSICS
            # on LM: 5ms, 100 * (iteration+1) steps, just use rotation
            solution = [[hypothesis]] + self.fixed
            self.simulator.initialize_solution(solution)

            # if config.FIX_OTHERS:
            #     for fix in self.fixed:
            #         self.simulator.fix_hypothesis(fix[0])
            # st = time.time()
            step_per_iter = self.PHYSICS_STEPS  # 20 on YCB, 100 on LM -- single hyp: *(iteration+1); multi hyp: constant
            steps = step_per_iter  # TODO step_per_iter * (iteration+1)
            T_phys = self.simulator.simulate_no_render(hypothesis.id, delta=1/60.0, steps=SIM_STEPS)  # 3 equivalent to 10x5ms (paper for YCBV) TODO was delta=0.005, steps=steps)
            # print("%0.1fms" % ((time.time() - st) * 1000))
            # compute displacement (0... high, 1... no displacement)
            # disp_t = 1.0 - min(np.linalg.norm(T_phys[:3, 3] - hypothesis.transformation[:3, 3]) / (TAU/1000), 1.0)
            # disp_q = max(0.0, np.dot(Rotation.from_dcm(T_phys[:3, :3]).as_quat(), Rotation.from_dcm(hypothesis.transformation[:3, :3]).as_quat()))

            # if config.FIX_OTHERS:
            #     self.simulator.unfix()

            # just use the rotation -- both are in camera system, so do not need to change the offset
            # TODO change in simulate() as well!
            hypothesis.transformation[:3, :3] = T_phys[:3, :3]  # only R
            # hypothesis.transformation = T_phys.copy()  # full trafo
            # hypothesis.confidence = disp_t * disp_q

            physics_hypotheses.append(hypothesis.duplicate())

            # === DF
            # st = time.time()

            hypothesis.refiner_param[6] = [Rotation.from_dcm(hypothesis.transformation[:3, :3]).as_quat(),
                                           np.array(hypothesis.transformation[:3, 3]).T[0], 1.0]
            # if unexplained is not None:
                # hypothesis.refiner_param[1] = hypothesis.refiner_param[1].copy()
                # hypothesis.refiner_param[1][unexplained==0] = 0
            hypothesis.refiner_param[10] = explained

            q, t, _ = self.refiner.refine(*hypothesis.refiner_param)
            hypothesis.transformation[:3, :3] = Rotation.from_quat(q).as_dcm()
            hypothesis.transformation[:3, 3] = t.reshape(3, 1)
            # hypothesis.confidence = c
            physics_hypotheses.append(hypothesis.duplicate())

            # print("%0.1fms" % ((time.time() - st) * 1000))
        return physics_hypotheses

    def refine_mitash(self, hypothesis, override_iterations=-1, explained=None):
        """
        like in Mitash et al. -> refine using all iterations, then physics, take full trafo from physics as result
        :param hypothesis:
        :return:
        """

        # TODO check if this is correct
        if override_iterations != -1:
            hypothesis.refiner_param[-4] = override_iterations

        self.simulator.objects_to_use = [hypothesis.id] + [fix[0].id for fix in self.fixed]
        self.simulator.reset_objects()

        # === DF
        hypothesis.refiner_param[6] = [Rotation.from_dcm(hypothesis.transformation[:3, :3]).as_quat(),
                                       np.array(hypothesis.transformation[:3, 3]).T[0], 1.0]
        hypothesis.refiner_param[10] = explained

        q, t, _ = self.refiner.refine(*hypothesis.refiner_param)
        hypothesis.transformation[:3, :3] = Rotation.from_quat(q).as_dcm()
        hypothesis.transformation[:3, 3] = t.reshape(3, 1)

        # === PHYSICS
        # on LM: 5ms, 100 * (iteration+1) steps, just use rotation
        solution = [[hypothesis]] + self.fixed
        self.simulator.initialize_solution(solution)

        T_phys = self.simulator.simulate_no_render(hypothesis.id, delta=1/60.0, steps=SIM_STEPS)
        hypothesis.transformation = T_phys.copy()  # full trafo

        return hypothesis

    def simulate(self, hypothesis):
        self.simulator.objects_to_use = [hypothesis.id] + [fix[0].id for fix in self.fixed]
        self.simulator.reset_objects()

        # === PHYSICS
        # on LM: 5ms, 100 * (iteration+1) steps, just use rotation
        solution = [[hypothesis]] + self.fixed
        self.simulator.initialize_solution(solution)

        # if config.FIX_OTHERS:
        #     for fix in self.fixed:
        #         self.simulator.fix_hypothesis(fix[0])

        T_phys = self.simulator.simulate_no_render(hypothesis.id, delta=1 / 60.0,
                                                   steps=SIM_STEPS)  # 3 are equivalent to 10x5ms (paper)

        # compute displacement (0... high, 1... no displacement) TODO consider all objects or just new one?
        max_displacement = 9.81 * (3/60.0)**2  # [m]
        disp_t = 1.0 - min(np.linalg.norm(T_phys[:3, 3] - hypothesis.transformation[:3, 3]) / max_displacement, 1.0)
        disp_q = max(0.0, np.dot(Rotation.from_dcm(T_phys[:3, :3]).as_quat(),
                                 Rotation.from_dcm(hypothesis.transformation[:3, :3]).as_quat()))

        return disp_q * disp_t, T_phys


# ================================================================
# ===== BUDGET ALLOCATION BANDIT (Object-level Verification) =====
# ================================================================

class BudgetAllocationBandit:

    """
    TODO
    """

    def __init__(self, pir, observation, hypotheses, unexplained=None):
        self.pir = pir
        self.observation = observation

        self.pool = [[h] for h in hypotheses]# + [None] * len(hypotheses)]  # for the phys hypotheses
        if not isinstance(self.pool[0], list):
            self.pool = [self.pool]
        self.max_iter = ITERATIONS * len(hypotheses) + len(hypotheses)#MAX_REFINEMENTS_PER_HYPOTHESIS + len(hypotheses)  # for initial play that is no refinement


        # # TODO debug hypotheses and fit computation
        # import matplotlib.pyplot as plt
        # def debug():
        #     fits = []
        #     for row in range(5):
        #         for col in range(5):
        #             i = col + (row * 5)
        #             fits.append(hypotheses[i].fit(observation, unexplained=unexplained))
        #     best_fits = np.argsort(fits)[::-1]
        #     for row in range(5):
        #         for col in range(5):
        #             i = col + (row * 5)
        #             j = best_fits[i]
        #             plt.subplot(5, 5, i + 1)
        #             vis = observation['depth'] - hypotheses[j].render(observation, 'depth')[1] * 1000
        #             vis[unexplained==0] = -1000
        #             plt.imshow(vis)
        #             plt.title("hi=%i, fit=%0.3f" % (j, hypotheses[j].fit(observation, unexplained=unexplained)))
        # # plt.show()
        # if int(hypotheses[0].model) == 1 and (unexplained==0).sum() > 0:
        #     drawnow(debug)
        #     plt.pause(5.0)

        # ---
        # INIT

        self.factor = 0.0  # e.g. 0.1 -> at most 0.1 change in reward due to stability -- 0 for DF, >0 for PCS

        self.arms = len(hypotheses)
        self.fits = np.zeros((self.arms, self.max_iter))#TODO was self.max_iter + 1))
        # self.pir.fixed = fixed
        for hi, h in enumerate(hypotheses):
            stability = 0.0#self.pir.simulate(h) * self.factor if self.factor > 0 else 0.0
            self.fits[hi, 0] = h.fit(observation, unexplained=unexplained)

            _, T_phys = self.pir.simulate(h)
            h_phys = h.duplicate()
            # h_phys.transformation[:3, :3] = T_phys[:3, :3]  # TODO change in refine as well
            h_phys.transformation = T_phys.copy()
            fit = h_phys.fit(observation, unexplained=unexplained)
            # self.pool[hi+len(hypotheses)][0] = h_phys
            # self.fits[hi+len(hypotheses), 0] = fit
            if fit > self.fits[hi, 0]:
                self.pool[hi] = [h_phys]
                self.fits[hi, 0] = fit

            self.pool[hi][0].confidence = self.fits[hi, 0]
        # self.pir.fixed = []
        self.fits[np.isnan(self.fits)] = 0

        self.plays = [1] * self.arms
        self.rewards = [fit for fit in self.fits[:, 0]]  # init reward with fit of initial hypothesis

        self.active = np.ones(self.arms)

        # n_removed = 0
        # for hi, fit in enumerate(self.fits[:, 0]):
        #     if fit < 0.05:
        #         del self.plays[hi-n_removed]
        #         del self.rewards[hi-n_removed]
        #         n_removed += 1
        # if len(self.plays) == 0:
        #     self.plays = [self.max_iter]

    # TODO caller has to handle the fixing of the environment
    def refine(self, fixed=[], unexplained=None):
        iteration = np.sum(self.plays)
        if iteration < self.max_iter:
            # SELECT
            c = C

            ucb_scores = [r + np.sqrt(c) * np.sqrt(np.log(iteration) / n) for r, n in zip(self.rewards, self.plays)]
            ucb_scores = np.array(ucb_scores)

            hi = np.argmax(ucb_scores)

            h = self.pool[hi][-1]  # most recent refinement of this hypothesis

            # EXPAND
            child = h.duplicate()

            # ROLLOUT
            self.pir.fixed = fixed
            physics_hypotheses = self.pir.refine(child, override_iterations=1, explained=fixed)
            _, T_phys = self.pir.simulate(child)
            physics_child = child.duplicate()
            physics_child.transformation = T_phys.copy()

            child = physics_hypotheses[-1]

            # REWARD  # TODO render fixed hypotheses as well? but cost only for object -- stimulate with fixed!
            reward = child.fit(self.observation, unexplained=unexplained)
            child.confidence = reward

            reward_phys = physics_child.fit(self.observation, unexplained=unexplained)
            physics_child.confidence = reward_phys

            if reward_phys > reward:
                reward = reward_phys
                child = physics_child

            # BACKPROPAGATE
            self.fits[hi, self.plays[hi]] = reward
            self.pool[hi].append(child)

            # running mean
            self.rewards[hi] = (self.rewards[hi] * float(self.plays[hi]) + reward) / (self.plays[hi] + 1.0)

            self.plays[hi] += 1
            self.pir.fixed = []  # reset s.t. at any time original state is retained

    def refine_max(self, fixed=[], unexplained=None):
        # global TAU
        for iteration in range(self.max_iter - np.sum(self.plays)):  # reduce by already played refinements
            # TAU -= 0.1
            self.refine(fixed, unexplained=unexplained)
        # TAU = 15

    def get_best(self):
        # # # TODO debug plot
        # def debug():
        #     best_n, fits_n = self.get_best_n(25)
        #     for hi, (h, fit) in enumerate(zip(best_n, fits_n)):
        #         plt.subplot(5, 5, hi+1)
        #         ren_d = h.render(self.observation, 'depth')[1] * 1000
        #         vis = np.abs(ren_d - self.observation['depth'])
        #         # vis[ren_d == 0] = 0
        #         # plt.imshow(vis[h.roi[0]:h.roi[2], h.roi[1]:h.roi[3]])
        #         plt.imshow(vis)
        #         plt.title("%0.2f" % fit)
        # drawnow(debug)
        # plt.pause(3.0)

        # select best fit and add it to final solution
        best_hi, best_ri = np.unravel_index(self.fits.argmax(), self.fits.shape)

        # print(self.fits)
        # print(self.pool[best_hi][best_ri].confidence)

        return self.pool[best_hi][best_ri], -1, -1#, self.plays[
            #best_hi % len(self.plays)] - 1, self.fits.max()  # do not count initial render

    def get_best_n(self, n):
        best_n = []
        fits_n = []
        fits = self.fits.copy()
        for i in range(min(n, (self.fits > 0).sum())):
            # select best fit and add it to final solution
            best_hi, best_ri = np.unravel_index(fits.argmax(), fits.shape)
            best_n.append(self.pool[best_hi][best_ri])

            fits_n.append(fits[best_hi, best_ri])
            fits[best_hi, best_ri] = -1
        return best_n, fits_n


# ================================================================
# ===== BUDGET ALLOCATION BANDIT (Scene-level Verification) =====
# ================================================================


class SceneBAB:

    """
    TODO
    """

    def __init__(self, pir, observation, hypotheses):
        self.pir = pir
        self.observation = observation

        self.pool = [[h] for h in hypotheses]# + [None] * len(hypotheses)]  # for the phys hypotheses
        if not isinstance(self.pool[0], list):
            self.pool = [self.pool]
        self.max_iter = ITERATIONS * len(hypotheses) + len(hypotheses)#MAX_REFINEMENTS_PER_HYPOTHESIS + len(hypotheses)  # for initial play that is no refinement

        # ---
        # INIT

        self.arms = len(hypotheses)
        self.fits = np.zeros((self.arms, 200))#self.max_iter))
        for hi, h in enumerate(hypotheses):
            self.fits[hi, 0] = h.fit(observation, unexplained=None)

            _, T_phys = self.pir.simulate(h)
            h_phys = h.duplicate()
            h_phys.transformation = T_phys.copy()
            fit = h_phys.fit(observation, unexplained=None)
            if fit > self.fits[hi, 0]:
                self.pool[hi] = [h_phys]
                self.fits[hi, 0] = fit

        self.plays = [1] * self.arms
        self.rewards = [fit for fit in self.fits[:, 0]]  # init reward with fit of initial hypothesis

        self.discounted_pulls = np.ones(self.arms)
        self.discounted_rewards = np.array(self.fits[:, 0])

    # TODO caller has to handle the fixing of the environment
    def refine(self, fixed=[], unexplained=None):
        # SELECT
        c = C

        # a) plain UCB
        # ucb_scores = [r + np.sqrt(c) * np.sqrt(np.log(np.sum(self.plays)) / n) for r, n in
        #               zip(self.rewards, self.plays)]
        # # b) discounted UCB
        eta = 1.0
        n_t_gamma = np.sum(self.discounted_pulls)
        assert n_t_gamma <= np.sum(self.plays)
        ucb_scores = [r/n + np.sqrt(c) * np.sqrt(eta * np.log(n_t_gamma) / n) for r, n in zip(self.discounted_rewards, self.discounted_pulls)]

        ucb_scores = np.array(ucb_scores)

        hi = np.argmax(ucb_scores)

        h = self.pool[hi][-1]  # most recent refinement of this hypothesis

        # EXPAND
        child = h.duplicate()

        # ROLLOUT
        self.pir.fixed = fixed
        physics_hypotheses = self.pir.refine(child, override_iterations=1, explained=fixed)
        _, T_phys = self.pir.simulate(child)
        physics_child = child.duplicate()
        physics_child.transformation = T_phys.copy()

        child = physics_hypotheses[-1]

        # MONITOR - take best fitting
        fit = child.fit(self.observation, unexplained=unexplained)
        child.confidence = fit

        fit_phys = physics_child.fit(self.observation, unexplained=unexplained)
        physics_child.confidence = fit_phys

        if fit_phys > fit:
            fit = fit_phys
            child = physics_child

        # BACKPROPAGATE
        self.fits[hi, self.plays[hi]] = fit
        self.pool[hi].append(child)

        # self.plays[hi%self.arms] += 1
        self.pir.fixed = []  # reset s.t. at any time original state is retained

        return hi, child, fit

    def backpropagate(self, hi, reward):
        # running mean
        self.rewards[hi] = (self.rewards[hi] * float(self.plays[hi]) + reward) / (self.plays[hi] + 1.0)
        self.plays[hi] += 1

        gamma = 0.99#1 - 0.25 * np.sqrt(T)
        self.discounted_pulls *= gamma
        self.discounted_rewards *= gamma
        self.discounted_pulls[hi] += 1
        self.discounted_rewards[hi] += reward# (self.discounted_rewards[hi] * float(self.discounted_pulls[hi]) + reward) / (self.discounted_pulls[hi] + 1.0)

    def get_best(self):
        # # # TODO debug plot
        # def debug():
        #     best_n, fits_n = self.get_best_n(25)
        #     for hi, (h, fit) in enumerate(zip(best_n, fits_n)):
        #         plt.subplot(5, 5, hi+1)
        #         ren_d = h.render(self.observation, 'depth')[1] * 1000
        #         vis = np.abs(ren_d - self.observation['depth'])
        #         # vis[ren_d == 0] = 0
        #         # plt.imshow(vis[h.roi[0]:h.roi[2], h.roi[1]:h.roi[3]])
        #         plt.imshow(vis)
        #         plt.title("%0.2f" % fit)
        # drawnow(debug)
        # plt.pause(3.0)

        # select best fit and add it to final solution
        best_hi, best_ri = np.unravel_index(self.fits.argmax(), self.fits.shape)

        return self.pool[best_hi][best_ri], -1, -1
            # self.plays[
            # best_hi % len(self.plays)] - 1, self.fits.max()  # do not count initial render

    def get_best_n(self, n):
        best_n = []
        fits_n = []
        fits = self.fits.copy()
        for i in range(min(n, (self.fits > 0).sum())):
            # select best fit and add it to final solution
            best_hi, best_ri = np.unravel_index(fits.argmax(), fits.shape)
            best_n.append(self.pool[best_hi][best_ri])

            fits_n.append(fits[best_hi, best_ri])
            fits[best_hi, best_ri] = -1
        return best_n, fits_n
