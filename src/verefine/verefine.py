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
TRACK_CONVERGENCE = True  # get best hypotheses at each iteration of SV (1 iter ~ [OV budget] iterations of plain MCTS)
# POLICY = UCB
C = 1  # TODO was 1e-1

ALL_OBJECTS = True
HYPOTHESES_PER_OBJECT = 25
ITERATIONS = 1
REFINEMENTS_PER_ITERATION = 1
BUDGET_SCALE = 1 if not ALL_OBJECTS else 3  # scales max iterations of verification (1... nobject * max iter per object -> one wrong node and we cannot refine all)
REFINE_AT_ONCE = True  # during SV, spend budget one-by-one (False) or all at once (True)

MAX_REFINEMENTS_PER_HYPOTHESIS = ITERATIONS * REFINEMENTS_PER_ITERATION * HYPOTHESES_PER_OBJECT  # BUDGET
MAX_MCTS_ITERATIONS_PER_OBJECT = MAX_REFINEMENTS_PER_HYPOTHESIS + HYPOTHESES_PER_OBJECT  # BUDGET + num hypotheses (initial)
USE_HEURISTIC_EXPANSION = False
DEBUG_PLOT = False
DEBUG_PLOT_REFINE = False
TAU, TAU_VIS = 20, 10  # [mm]

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
    depth_obs = observation['depth']
    depth_ren = rendered[1] * 1000  # in mm TODO or do this in renderer?

    # mask = np.logical_and(mask, depth_ren>0)#np.logical_and(mask, np.logical_and(depth_ren > 0, depth_obs > 0))# TODO or just
    mask = depth_ren>0#np.logical_and(unexplained, depth_ren>0)
    if np.count_nonzero(mask) == 0:  # no valid depth values
        cost_durations.append(time.time() - st)
        return 0

    # masks
    depth_mask = np.logical_and(mask, depth_ren - depth_obs < TAU_VIS)  # only visible -- ren at most [TAU_VIS] behind obs
    cos_mask = depth_mask#np.logical_and(mask, depth_ren - depth_obs < 10)

    # visibility
    visibility_ratio = float(depth_mask.sum()) / float((depth_ren > 0).sum())  # visible / rendered count

    # delta fit
    dist = np.abs(depth_obs[depth_mask] - depth_ren[depth_mask])
    delta = np.mean(np.min(np.vstack((dist / TAU, np.ones(dist.shape[0]))), axis=0))

    # # cos fit
    # norm_obs = observation['normals']
    # norm_ren = rendered[3]
    # on = norm_obs[cos_mask].reshape(cos_mask.sum(), 3)
    # rn = norm_ren[cos_mask].reshape(cos_mask.sum(), 3)
    # cos = np.einsum('ij,ij->i', on, rn)
    # cos[cos < 0] = 0  # clamp to zero
    # cos_fit = np.mean(cos)  # TODO or also linear until threshold?
    #
    # # Aldoma
    # assert depth_mask.shape[0] == cos_mask.shape[0]
    # aldoma_fit = (1 - dist/TAU) * cos
    # aldoma_fit[dist > TAU] = 0
    # aldoma_fit = np.mean(aldoma_fit)

    # TODO multi assignment + scaling per term
    # fit = visibility_ratio * aldoma_fit
    # fit = visibility_ratio * ((1 - delta)*0.7 + cos_fit*0.3)
    # fit = visibility_ratio * (1 - delta) * cos_fit
    if unexplained is None or float((depth_mask>0).sum()) == 0:
        overlap = 0
    else:
        overlap = float(np.logical_and(unexplained==0, depth_mask>0).sum()) / float((depth_mask>0).sum())#float(np.logical_or(unexplained==0, depth_ren>0).sum())
    fit = visibility_ratio * (1 - delta) * (1-overlap)
    if np.isnan(fit):  # invisible object
        cost_durations.append(time.time() - st)
        return 0

    cost_durations.append(time.time() - st)
    return fit


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

        assert mode in ['color', 'depth', 'depth+seg', 'depth+normal', 'color+depth+seg', 'cost']

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

        # a) render depth, compute score on CPU
        rendered = self.render(observation, mode='depth')#+normal')
        # unexplained = np.ones_like(self.mask)# TODO self.mask
        # for hs in fixed:
        #     h_mask = hs[0].render(observation, mode='depth+seg')[2]
        #     unexplained[h_mask > 0] = 0
        score = fit(observation, rendered, unexplained)

        # b) render depth, compute score on GPU  TODO has a bug when object is too far off
        # _, score = self.render(observation, "cost", fixed)

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

    def refine(self, hypothesis, override_iterations=-1, unexplained=None):
        """

        :param hypothesis:
        :return:
        """

        iterations = hypothesis.refiner_param[-3] if override_iterations == -1 else override_iterations
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
            T_phys = self.simulator.simulate_no_render(hypothesis.id, delta=1/60.0, steps=60)  # 3 equivalent to 10x5ms (paper for YCBV) TODO was delta=0.005, steps=steps)
            # print("%0.1fms" % ((time.time() - st) * 1000))
            # compute displacement (0... high, 1... no displacement)
            # disp_t = 1.0 - min(np.linalg.norm(T_phys[:3, 3] - hypothesis.transformation[:3, 3]) / (TAU/1000), 1.0)
            # disp_q = max(0.0, np.dot(Rotation.from_dcm(T_phys[:3, :3]).as_quat(), Rotation.from_dcm(hypothesis.transformation[:3, :3]).as_quat()))

            # if config.FIX_OTHERS:
            #     self.simulator.unfix()

            # just use the rotation -- both are in camera system, so do not need to change the offset
            hypothesis.transformation[:3, :3] = T_phys[:3, :3]  # only R
            # hypothesis.transformation = T_phys.copy()
            # hypothesis.confidence = disp_t * disp_q

            physics_hypotheses.append(hypothesis.duplicate())

            # === DF
            # st = time.time()

            hypothesis.refiner_param[6] = [Rotation.from_dcm(hypothesis.transformation[:3, :3]).as_quat(),
                                           np.array(hypothesis.transformation[:3, 3]).T[0], 1.0]
            if unexplained is not None:
                hypothesis.refiner_param[1] = hypothesis.refiner_param[1].copy()
                hypothesis.refiner_param[1][unexplained==0] = 0
            q, t, _ = self.refiner.refine(*hypothesis.refiner_param)
            hypothesis.transformation[:3, :3] = Rotation.from_quat(q).as_dcm()
            hypothesis.transformation[:3, 3] = t.reshape(3, 1)
            # hypothesis.confidence = c
            physics_hypotheses.append(hypothesis.duplicate())

            # print("%0.1fms" % ((time.time() - st) * 1000))
        return physics_hypotheses

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
                                                   steps=3)  # 3 are equivalent to 10x5ms (paper)

        # compute displacement (0... high, 1... no displacement) TODO consider all objects or just new one?
        max_displacement = 9.81 * (3/60.0)**2  # [m]
        disp_t = 1.0 - min(np.linalg.norm(T_phys[:3, 3] - hypothesis.transformation[:3, 3]) / max_displacement, 1.0)
        disp_q = max(0.0, np.dot(Rotation.from_dcm(T_phys[:3, :3]).as_quat(),
                                 Rotation.from_dcm(hypothesis.transformation[:3, :3]).as_quat()))

        return disp_q * disp_t


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

        self.pool = [[h] for h in hypotheses + [None] * len(hypotheses)]  # for the phys hypotheses
        self.max_iter = 2 * len(hypotheses)#MAX_REFINEMENTS_PER_HYPOTHESIS + len(hypotheses)  # for initial play that is no refinement


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

        arms = len(hypotheses)
        self.fits = np.zeros((arms * 2, self.max_iter + 1))
        # self.pir.fixed = fixed
        for hi, h in enumerate(hypotheses):
            stability = 0.0#self.pir.simulate(h) * self.factor if self.factor > 0 else 0.0
            self.fits[hi] = h.fit(observation, unexplained=unexplained) * (1.0 - self.factor) + stability
        # self.pir.fixed = []

        self.plays = [1] * arms
        self.rewards = [fit for fit in self.fits[:, 0]]  # init reward with fit of initial hypothesis

        self.active = np.ones(arms)

    # TODO caller has to handle the fixing of the environment
    def refine(self, fixed=[], unexplained=None):
        iteration = np.sum(self.plays)
        if iteration < self.max_iter and self.active.sum() > 0:
            # SELECT
            c = 1  # TODO used 1e-3 for YCBV
            ucb_scores = [r + np.sqrt(c) * np.sqrt(np.log(iteration) / n) for r, n in zip(self.rewards, self.plays)]

            ucb_scores = np.array(ucb_scores)
            ucb_scores[self.active==0] = -1

            hi = np.argmax(ucb_scores)

            h = self.pool[hi][-1]  # most recent refinement of this hypothesis

            # EXPAND
            child = h.duplicate()

            # ROLLOUT
            self.pir.fixed = fixed
            # SIMULATOR.objects_to_use.append(child.id)
            physics_hypotheses = self.pir.refine(child, override_iterations=1, unexplained=unexplained)
            # SIMULATOR.objects_to_use = SIMULATOR.objects_to_use[:-1]
            physics_child = physics_hypotheses[-2]  # after last refinement step TODO was [0] - bc child was overwritten in refine?
            child = physics_hypotheses[-1]

            # REWARD  # TODO render fixed hypotheses as well? but cost only for object -- stimulate with fixed!
            stability = 0.0 #self.pir.simulate(child) * self.factor if self.factor > 0 else 0.0
            reward = child.fit(self.observation, unexplained=unexplained) * (1.0 - self.factor) + stability
            child.confidence = reward

            stability_phys = 0.0#self.pir.simulate(physics_child) * self.factor if self.factor > 0 else 0.0
            reward_phys = physics_child.fit(self.observation, unexplained=unexplained) * (1.0 - self.factor) + stability_phys
            physics_child.confidence = reward_phys

            # BACKPROPAGATE
            self.fits[hi][self.plays[hi]] = reward
            self.fits[hi + len(self.plays)][self.plays[hi]] = reward_phys
            self.pool[hi].append(child)  # hypothesis after physics + refine
            self.pool[hi + len(self.plays)].append(physics_child)  # hypothesis after physics

            self.rewards[hi] = (self.rewards[hi] * float(self.plays[hi]) + reward) / (
                        self.plays[hi] + 1.0)  # running mean

            self.plays[hi] += 1
            self.pir.fixed = []  # reset s.t. at any time original state is retained

            if self.fits[hi][self.plays[hi]-1] - self.fits[hi][self.plays[hi]-2] <= 0:
                self.active[hi] = 0

    def refine_max(self, fixed=[], unexplained=None):
        for iteration in range(self.max_iter - np.sum(self.plays)):  # reduce by already played refinements
            self.refine(fixed, unexplained=unexplained)

    def get_best(self):
        # select best fit and add it to final solution
        best_hi, best_ri = np.unravel_index(self.fits.argmax(), self.fits.shape)

        return self.pool[best_hi][best_ri], self.plays[
            best_hi % len(self.plays)] - 1, self.fits.max()  # do not count initial render


# =========================================
# ===== MCTS Scene-level Verification =====
# =========================================
import time
duration_select, duration_expand, duration_rollout, duration_backprop = [], [], [], []


def verefine_solution(hypotheses_pool):
    """
    :param hypotheses_pool: dict with (obj_id, list of Hypothesis)
    :return:
    """

    """
    CLUSTER
    """
    # if ALL_OBJECTS:
    #     hypotheses_pools, adjacencies = cluster(hypotheses_pool)
    #     # print("  %i cluster(s)" % len(hypotheses_pools))
    #
    #     # TODO debug -- all in one cluster
    #     # hypotheses_pools = [hypotheses_pool]
    # else:
    #     # TODO debug -- this should yield same result as verfine_individual
    #     hypotheses_pools = []
    #     adjacencies = []
    #     for obj_str, obj_hypotheses in hypotheses_pool.items():
    #         hypotheses_pools.append({obj_str: obj_hypotheses})
    #         adjacencies.append({})

    # TODO debug - no clustering
    hypotheses_pools = [hypotheses_pool]
    adjacencies = [dict()]
    all_obj_strs = list(hypotheses_pool.keys())
    for obj_str in all_obj_strs:
        adjacencies[0][obj_str] = all_obj_strs

    """
    VEReFINE
    """

    # TODO where to put this?
    SceneVerificationNode.best_per_object = {}
    SceneVerificationNode.best_scene = [-1, None]

    def print_tree(node):
        if node is not None:
            print(node)
            for child in node.children:
                print_tree(child)

    def best_per_object(node):
        if node is None:
            return
        if node.refiner is not None:
            h, plays, fit = node.refiner.get_best()
            if fit > best_fit_per_object[h.model]:
                best_fit_per_object[h.model] = fit
                best_h_per_object[h.model] = h

        for child in node.children:
            h, plays, fit = child.refiner.get_best()
            if fit > best_fit_per_object[h.model]:
                best_fit_per_object[h.model] = fit
                best_h_per_object[h.model] = h
            best_per_object(child)

    def best_scene(node, best_reward=-1):
        if node is None:
            return -1, None

        best_node = node

        # # a) highest reward
        # for child in node.children:
        #     if child.best_reward > best_reward:  # TODO child.reward or child.best_reward?
        #         best_reward, best_node = best_scene(child, child.reward)

        # b) most-played path until end
        if len(node.children) > 0:
            most_played, most_plays = None, -1
            for child in node.children:
                if child.plays > most_plays:
                    most_plays = child.plays
                    most_played = child
            best_node = best_scene(most_played)[1]

        # # c) best-reward path until end
        # if len(node.children) > 0:
        #     most_played, most_plays = None, -1
        #     for child in node.children:
        #         if child.best_reward > most_plays:
        #             most_plays = child.best_reward
        #             most_played = child
        #     best_node = best_scene(most_played)[1]

        return best_node.reward, best_node

    convergence_hypotheses = {}
    final_hypotheses = []
    final_reward = 0
    for hypotheses_pool, adjacency in zip(hypotheses_pools, adjacencies):
        initial_solution = VerificationSolution(candidates=(hypotheses_pool, adjacency))
        root = SceneVerificationNode(initial_solution)

        all_obj_strs = list(hypotheses_pool.keys())
        nobjects = len(hypotheses_pool.keys())
        max_iterations = 7#int(min(MAX_MCTS_ITERATIONS_PER_OBJECT * nobjects * BUDGET_SCALE, 300))
        if DEBUG_PLOT:
            print("   - #obj_initial = %i" % nobjects)
            print("   - verifine for %i iterations..." % (max_iterations))
        for iteration in range(max_iterations):
            # --- SELECTION - apply selection policy as long as we have the statistics (i.e., fully expanded)
            st = time.time()
            selected_node = root.select()  #
            duration_select.append(time.time()-st)

            # --- EXPANSION - apply expansion policy to get new node
            st = time.time()
            new_node = selected_node.expand(use_heuristic=USE_HEURISTIC_EXPANSION)
            duration_expand.append(time.time() - st)

            # --- ROLLOUT - nested bandit to generate rollout policy - compute scene reward for tree
            st = time.time()
            reward = new_node.rollout()
            duration_rollout.append(time.time() - st)
            # print(duration_rollout[-1])

            # --- BACKPROPAGATION - update statistics of nodes that we descended through
            st = time.time()
            new_node.backpropagate(reward)
            duration_backprop.append(time.time() - st)

            # --- DEBUG - analyze convergence
            if TRACK_CONVERGENCE:

                best_fit_per_object = {}
                best_h_per_object = {}
                for obj_str in list(hypotheses_pool.keys()):
                    best_fit_per_object[obj_str] = -1
                    best_h_per_object[obj_str] = None
                #
                # # a) gather best hypotheses (per object) in tree
                # best_per_object(root)

                # b) best scene
                best_scene_reward, best_scene_node = best_scene(root)  # SceneVerificationNode.best_scene
                while best_scene_node.parent is not None:
                    cur_hyp, _, cur_fit = best_scene_node.refiner.get_best()
                    best_h_per_object[cur_hyp.model] = cur_hyp
                    best_fit_per_object[cur_hyp.model] = cur_fit

                    best_scene_node = best_scene_node.parent

                # to old format + add missing objects
                if not iteration in convergence_hypotheses:
                    convergence_hypotheses[iteration] = []
                for obj_str, h_best in best_h_per_object.items():
                    # # TODO missing object; refine it without considering other objects
                    if h_best is None and obj_str in SceneVerificationNode.best_per_object:
                        h_best = SceneVerificationNode.best_per_object[obj_str][
                            1]  # TODO could also check [0] to reject bad fit

                    if h_best is not None:
                        convergence_hypotheses[iteration].append([h_best])

        if DEBUG_PLOT:
            print("%s, %0.3f" % tuple(root.get_best()))

        # === B) get best fit per object
        best_fit_per_object = {}
        best_h_per_object = {}
        for obj_str in list(hypotheses_pool.keys()):
            best_fit_per_object[obj_str] = -1
            best_h_per_object[obj_str] = None
        #
        # # a) gather best hypotheses (per object) in tree
        # best_per_object(root)

        # b) best scene
        best_scene_reward, best_scene_node = best_scene(root)  # SceneVerificationNode.best_scene
        while best_scene_node.parent is not None:
            cur_hyp, _, cur_fit = best_scene_node.refiner.get_best()
            best_h_per_object[cur_hyp.model] = cur_hyp
            best_fit_per_object[cur_hyp.model] = cur_fit

            best_scene_node = best_scene_node.parent

        # to old format + add missing objects
        cluster_hypotheses = []
        for obj_str, h_best in best_h_per_object.items():
            # # TODO missing object; refine it without considering other objects
            if h_best is None:
                # refiner = BudgetAllocationBandit(REFINER, OBSERVATION, hypotheses_pool[obj_str])
                # for iteration in range(ITERATIONS):
                #     refiner.refine()
                # h_best = refiner.get_best()[0]
                if obj_str in SceneVerificationNode.best_per_object:
                    # if SceneVerificationNode.best_per_object[obj_str][0] > 0.1:  # TODO reject bad fit?
                    h_best = SceneVerificationNode.best_per_object[obj_str][1]

            if h_best is not None:
                cluster_hypotheses.append([h_best])

        # ===
        if DEBUG_PLOT:
            print("   - #obj_verifine = %i\n" % len(cluster_hypotheses))

        final_hypotheses += cluster_hypotheses
        final_reward += reward  # TODO actually would have to render scene with all clusters and compute overall reward

    return final_hypotheses, final_reward, convergence_hypotheses


class SceneVerificationNode:

    """
    Node of search tree
    """

    best_scene = [-1, None]
    best_per_object = {}

    def __init__(self, solution, parent=None):
        self.solution = solution

        # -- tree
        self.parent = parent  # None if root
        self.children = []  # expand adds new nodes to this list
        self.fully_expanded = False  # flag is set by first call to expand that gets no candidates
        self.final = False  # flag is set by first call to expand where solution has no further candidates

        # -- stats
        self.plays = 0  # total number of rollouts that this node was involved in
        self.reward = 0  # mean reward over all rollouts
        self.best_reward = 0

        # -- selection policy
        if self.parent is None:
            self.refiner = None  # TODO refiner could manage plane hypotheses
        else:
            self.refiner = solution.selected[solution.selection_order[-1]]

        if len(self.solution.candidates[0]) == 0:
            self.fully_expanded = True
            self.final = True

    def __str__(self):
        return "object %s" % self.solution.selection_order  # TODO per object in selection order, get best hypothesis

    def select(self):
        """
        non-stationary multi-armed bandit policy -> SMPyBandits
        TODO allow to use different policies
        :return selected Node
        """
        if self.fully_expanded and not self.final:
            ucb_scores = [child.reward + np.sqrt(C) * np.sqrt(np.log(self.plays) / child.plays)
                          for child in self.children]
            selected = self.children[np.argmax(ucb_scores)]

            # recursively call child to select its best child using selection policy
            return selected.select()
        else:
            return self

    def expand(self, use_heuristic=False):
        """
        expansion should offer...
        a) uniform
        b) heuristic
        options (for evaluation of the heuristic)

        :return self if fully expanded; else the newly created node
        """
        new_node = self

        if not self.fully_expanded:
            candidates = self._get_candidates()

            # uniformly sample a new child node and add it to the tree
            # np.random.seed(seed)
            child_id = np.random.choice(candidates)
            child = SceneVerificationNode(self.solution.expand(child_id), parent=self)
            self.children.append(child)

            new_node = child

            if len(self._get_candidates()) == 0:
                # in this case we apply selection policy in the next iteration
                self.fully_expanded = True

        return new_node

    def rollout(self):
        """
        in selection order...
        - fix previous objects (in the hypothesis given by their refinement bandit)
        - call refinement bandit (-> refinement bandit selects which hypotheses to choose during rollout -> sets policy)
        """

        # compute descent path TODO or let selection track this and just pass it to rollout?
        descent_path = []
        current = self
        while current is not None:
            descent_path.append(current)
            current = current.parent
        descent_path = descent_path[::-1]

        # 1) get best hypothesis for each object in selection order...
        selected_hypotheses = []
        SIMULATOR.objects_to_use = []
        SIMULATOR.reset_objects()
        # TODO fix everything - just unfix the current hypothesis
        unexplained = None#np.ones((480, 640), dtype=np.uint8)
        for current in descent_path:

            if current.refiner is None:
                assert current.parent is None  # this should only be the case for the root -- TODO unless we add a refiner for pose hypotheses
                continue  # TODO then we remove this continue

            # === REFINER
            # best_ri = np.argmax(np.hstack((current.refiner.fits[:25, :], current.refiner.fits[25:, :])), axis=1)
            # hypotheses = [current.refiner.pool[hi + int(25*np.floor(ri/25))][ri%25] for hi, ri in enumerate(best_ri)]
            # current.refiner = BudgetAllocationBandit(current.refiner.pir, current.refiner.observation, hypotheses, unexplained=unexplained)
            if REFINE_AT_ONCE:
                current.refiner.refine_max(fixed=selected_hypotheses, unexplained=unexplained)  # TODO is this already covered by adding fix_hypothesis and unfix below?
            else:  # only one iteration
                current.refiner.refine(fixed=selected_hypotheses, unexplained=unexplained)

            # === update fixed objects
            # get best hypothesis up till now
            best_hypothesis, _, best_fit = current.refiner.get_best()

            # keep track of best hypothesis per object
            if best_hypothesis.model not in SceneVerificationNode.best_per_object:
                SceneVerificationNode.best_per_object[best_hypothesis.model] = [best_fit, best_hypothesis]
            elif SceneVerificationNode.best_per_object[best_hypothesis.model][0] < best_fit:
                SceneVerificationNode.best_per_object[best_hypothesis.model] = [best_fit, best_hypothesis]

            # # fix this object (with best hypothesis) in simulation for further refinements in descent path
            # SIMULATOR.fix_hypothesis(best_hypothesis)
            # SIMULATOR.objects_to_use.append(best_hypothesis.id)

            # store for final solution
            selected_hypotheses.append([best_hypothesis])

            # selected_depth = best_hypothesis.render(OBSERVATION, 'depth')[1]
            # unexplained[selected_depth > 0] = 0

            # # TODO debug
            # obj_ids = [RENDERER.dataset.objlist.index(int(obj_hs[0].id[:2])) for obj_hs in
            #            selected_hypotheses]  # TODO do this conversion in renderer
            # obj_trafos = [obj_hs[0].transformation for obj_hs in selected_hypotheses]
            # rendered = RENDERER.render(obj_ids, obj_trafos,
            #                            OBSERVATION['extrinsics'], OBSERVATION['intrinsics'],
            #                            mode='color')

            # def debug():
            #     plt.subplot(1, 2, 1)
            #     plt.imshow(rendered[0] / 255 * 0.7 + OBSERVATION['rgb'] / 255 * 0.3)
            #     plt.subplot(1, 2, 2)
            #     plt.imshow(unexplained)
            # drawnow(debug)
            # plt.pause(0.5)
        # unfix hypotheses
        # SIMULATOR.unfix()

        # 2) randomly select an hypothesis for unplaced objects, simulate the scene with the known ones fixed
        # -- get list of all objects in cluster
        root = self
        while root.parent is not None:
            root = root.parent
        all_objects = root.solution.initial_candidates[0]
        # --- randomly select one hypothesis of the missing objects
        selected_names = [obj_hs[0].model for obj_hs in selected_hypotheses]
        # a)
        # np.random.seed(seed)
        missing_hypotheses = [[np.random.choice(obj_hyps)] for obj_name, obj_hyps in all_objects.items() if
                              obj_name not in selected_names]  # TODO only if fit of missing is good enough? otherwise crap may bias search

        # b) TODO useful? take best known hypothesis of missing objects -- if useful, could also do scoring of initial candidates and initialize best_per_object accordingly
        # missing_hypotheses = []
        # for obj_name, obj_hyps in all_objects.items():
        #     if obj_name in SceneVerificationNode.best_per_object:
        #         if SceneVerificationNode.best_per_object[obj_name][0] > 0.1:  # TODO only if fit is good enough?
        #             missing_hypotheses.append([SceneVerificationNode.best_per_object[obj_name][1]])
        #     else:
        #         missing_hypotheses.append([np.random.choice(obj_hyps)])

        # --- add to already fixed hypotheses
        scene_hypotheses = selected_hypotheses + missing_hypotheses

        # # TODO debug
        # obj_ids = [RENDERER.dataset.objlist.index(int(obj_hs[0].id[:2])) for obj_hs in
        #            selected_hypotheses]  # TODO do this conversion in renderer
        # obj_trafos = [obj_hs[0].transformation for obj_hs in selected_hypotheses]
        # selected_rgb = RENDERER.render(obj_ids, obj_trafos,
        #                            OBSERVATION['extrinsics'], OBSERVATION['intrinsics'],
        #                            mode='color')[0]

        # 3) render resulting scene and compute fitness
        obj_ids = [RENDERER.dataset.objlist.index(int(obj_hs[0].id[:2])) for obj_hs in scene_hypotheses]  # TODO do this conversion in renderer
        obj_trafos = [obj_hs[0].transformation for obj_hs in scene_hypotheses]
        # TODO just pass a list of hypotheses alternatively

        # # a) render depth, compute score on CPU
        rendered = RENDERER.render(obj_ids, obj_trafos,
                                   OBSERVATION['extrinsics'], OBSERVATION['intrinsics'],
                                   mode='depth+normal')
        scene_reward = fit(OBSERVATION, rendered, unexplained=np.ones_like(rendered[1], dtype=np.uint8))

        # def debug():
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(selected_rgb/255*0.7 + OBSERVATION['rgb']/255*0.3)
        #     plt.title(str(self) + " (%i plays)" % self.plays)
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(rendered[1])
        #     plt.title("final (reward=%0.3f)" % scene_reward)
        # drawnow(debug)
        # plt.pause(1.0)

        # # b) render depth, compute score on GPU
        # cost_id = None  # consider fit of full scene
        # # only consider fit of selected objects -> but considers occlusion
        # # cost_id = 0
        # # for v in [RENDERER.dataset.objlist.index(int(obj_hs[0].id[:2])) for obj_hs in selected_hypotheses]:  #obj_ids:
        # #     cost_id |= (1 << v)
        # rendered, scene_reward = RENDERER.render(obj_ids, obj_trafos,
        #                            OBSERVATION['extrinsics'], OBSERVATION['intrinsics'],
        #                            mode='cost', cost_id=cost_id)

        # keep track of best scene TODO this is more expensive than finding this out afterwards
        # if SceneVerificationNode.best_scene[0] < scene_reward:
        #     SceneVerificationNode.best_scene = [scene_reward, self]

        return scene_reward

    def backpropagate(self, reward, child=None):
        """
        update reward along selection path, i.e., backpropagate to parent until root is reached
        """

        if child is None:
            self.latest_reward = reward
            self.best_reward = self.best_reward if reward < self.best_reward else reward

        self.reward = (self.reward * float(self.plays) + reward) / (self.plays + 1.0)  # running mean
        self.plays += 1

        if self.parent is not None:  # i.e., this is not the root
            self.parent.backpropagate(reward, self)  # -> follows descent path and stops at root

    def get_best(self):
        if len(self.children) > 0:
            # if isinstance(self, RefinementNode):
                # TODO only for refinement -- would actually make sense for both -- we want the best rendering fit
            best = self
            best_reward = self.best_reward
            # else:
            #     best = None
            #     best_reward = -1

            for child in self.children:
                child_best, child_reward = child.get_best()
                if child_reward >= best_reward:
                    best = child_best
                    best_reward = child_reward
        else:
            best = self
            best_reward = self.best_reward  # TODO best, latest or mean reward? -> maybe best as it captures the fit of the hypothesis to the observation -- so mean reward would make no sense

        return best, best_reward

    def _get_candidates(self):
        """
        :return: list of candidate indices -- these can be used to call __expand_candidate
        """
        return self.solution.get_candidates()


class VerificationSolution:

    """
    TODO
    """

    def __init__(self, candidates=({},{}), selected={}, selection_order=[]):
        self.initial_candidates = candidates[0].copy(), candidates[1].copy()
        self.candidates = candidates[0].copy(), candidates[1].copy()  # initial hypotheses per object (key: obj_id, value: List[Hypothesis])
        self.selected = selected  # selected objects with their RefinementSolution (obj_id, RefinementSolution)
        self.selection_order = selection_order  # obj_ids, starting from root

        # remove all candidates that would fall (too far)
        self.verify_candidates()

    def get_candidates(self):
        """
        :return: list of possible obj_ids to select for subsequent solution
        """
        return list(self.candidates[0].keys())
        # return self.verify_candidates()  # TODO or do this each frame? then we cannot just pop candidates but must handle sublists

    def verify_candidates(self):

        # if len(self.selection_order) > 0:
        #     if self.selection_order[0] == '19':
        #         a = 1

        # print("-> selected %s" % self.selection_order[-1])
        # adjacency = list(self.candidates[0].keys())  # TODO could actually compute this for floor as well - then we even save the drop test
        # if EXPAND_ADJACENT:
        #     if len(self.selection_order) > 0 and len(self.candidates[1].keys()) > 0:
        #
        #         adjacency = []
        #         for selected_obj in self.selection_order:
        #             # check if they are adjacent to already placed objects
        #             adjacency += self.candidates[1][selected_obj]
        #
        #         # print("-> selected %s" % self.selection_order[-1])
        #         # print("---> candidates are %s" % adjacency)
        #
        # # get candidates
        # candidates = []
        # outliers = []
        # for hi, (obj_id, obj_hs) in enumerate(self.candidates[0].items()):
        #     if obj_id in adjacency or adjacency is None:
        #         h = obj_hs[0]  # TODO only take max-conf one? or current best? or use all and select most stable?
        #         candidates.append(h)
        #     else:
        #         outliers.append(obj_id)
        # for obj_id in outliers:
        #     self.candidates[0].pop(obj_id)
        #
        # if len(candidates) == 0:
        #     return
        #
        # # fix current base in its best poses
        # fixed = []
        # fixed_ids = []
        # for obj_id, refiner in self.selected.items():
        #     h = refiner.get_best()[0]
        #     fixed_ids.append(h.id)
        #     fixed.append(h)
        #     self.init_for_simulation(h, mass=0, collision_group=1, collision_mask=1)#base_mask)
        #
        # displacements = []
        # for hi, h in enumerate(candidates):
        #     if h is not None:
        #         SIMULATOR.objects_to_use = [h.id] + fixed_ids
        #         SIMULATOR.reset_objects()
        #
        #         # TODO why does this not work with all at once in different collision groups?
        #         self.init_for_simulation(h, mass=1, collision_group=2 + hi, collision_mask=1)
        #         ds, Ts = self.get_displacements([h], delta=0.01, steps=50)
        #         displacements.append(ds[0])
        #
        # # reject candidates with a too high displacement
        # th = 0.02
        # inliers = np.array(displacements) <= th  # TODO also filter-out obvious FPs (rendering fit < 0.1)
        #
        # # remove outliers
        # if inliers.sum() == 0:  # s.t. we do not throw away everything
        #     inliers[np.argmin(displacements)] = True
        # for candidate, is_inlier in zip(candidates, inliers):
        #     if not is_inlier:
        #         self.candidates[0].pop(candidate.model)
        pass

    def init_for_simulation(self, h, mass, collision_group, collision_mask):
        obj_str = h.id

        # save trafo for rendering
        T_obj = h.transformation.copy()
        # trafos[obj_str] = T_obj.copy()

        # apply transformation
        T_obj = SIMULATOR.T_gl * T_obj  # c2w * m2c
        R = T_obj[0:3, 0:3]

        # move object in simulation
        position = T_obj[0:3, 3] + (R * np.array([[0], [0], [SIMULATOR.dataset.obj_coms[int(obj_str[:2]) - 1]]]))
        orientation = Rotation.from_dcm(R).as_quat()
        SIMULATOR.pyb.resetBasePositionAndOrientation(SIMULATOR.models[obj_str], position, orientation,
                                                      SIMULATOR.world)  # also sets v to 0

        # 0... fix, >0... dynamic
        SIMULATOR.pyb.changeDynamics(SIMULATOR.models[obj_str], -1, mass=mass)
        # collide with candidates
        SIMULATOR.pyb.setCollisionFilterGroupMask(SIMULATOR.models[obj_str], -1, collision_group, collision_mask)

    def get_displacements(self, hs, delta=0.01, steps=10):
        # set-up simulation
        SIMULATOR.pyb.resetBasePositionAndOrientation(SIMULATOR.planeId, [0, 0, 0], [0, 0, 0, 1], SIMULATOR.world)
        SIMULATOR.pyb.setCollisionFilterGroupMask(SIMULATOR.planeId, -1, 1, 1)
        SIMULATOR.pyb.setTimeStep(delta)
        SIMULATOR.pyb.setPhysicsEngineParameter(fixedTimeStep=delta, numSolverIterations=20, numSubSteps=0)

        # simulate
        for i in range(steps):
            SIMULATOR.pyb.stepSimulation()

        # read-back transformations after simulation and compute displacements
        displacements = []
        Ts_after = []
        for h in hs:
            obj_str = h.id

            # 1) initial transformation
            T_before = h.transformation.copy()
            T_before = SIMULATOR.T_gl * T_before  # c2w * m2c

            # 2) simulated transformation
            position, orientation = SIMULATOR.pyb.getBasePositionAndOrientation(SIMULATOR.models[obj_str],
                                                                                SIMULATOR.world)
            position = list(position)
            orientation = list(orientation)

            T_after = np.matrix(np.eye(4))
            T_after[0:3, 0:3] = Rotation.from_quat(orientation).as_dcm()
            T_after[0:3, 3] = np.matrix(position).T \
                        - (T_after[0:3, 0:3] * np.array([[0], [0], [SIMULATOR.dataset.obj_coms[int(obj_str[:2]) - 1]]]))

            Ts_after.append(SIMULATOR.T_cv * T_after)
            # 3) compute displacement
            displacement = np.linalg.norm(T_before[:3, 3] - T_after[:3, 3])  # TODO or just z? or also consider R?
            displacements.append(displacement)
        return np.array(displacements), Ts_after

    def expand(self, obj_id):
        """
        creates a new VerificationSolution with this object selected and a corresponding RefinementSolution object
        """
        # remove object from candidates
        candidate_hypotheses = self.candidates[0].pop(obj_id, None)
        if candidate_hypotheses is None:
            raise ValueError("%i is no candidate object" % obj_id)
        # remove self from future candidates
        new_candidates = self.initial_candidates[0].copy(), self.initial_candidates[1].copy()
        new_candidates[0].pop(obj_id, None)

        # add object (and RefinementSolution) to selected
        # refinement_solution = RefinementSolution(candidate_hypotheses.copy())
        fixed = [[s.get_best()[0]] for s in list(self.selected.values())]
        # unexplained = np.ones_like(OBSERVATION['depth'], dtype=np.uint8)
        # for fix in fixed:
        #     unexplained[fix[0].render(OBSERVATION, 'depth')[1] > 0] = 0
        unexplained = None
        refinement_solution = BudgetAllocationBandit(REFINER, OBSERVATION, candidate_hypotheses,
                                                     unexplained=unexplained)
        new_selected = self.selected.copy()
        new_selected[obj_id] = refinement_solution
        new_selection_order = self.selection_order.copy()
        new_selection_order.append(obj_id)

        # return new VerificationSolution
        return VerificationSolution(candidates=new_candidates,
                                    selected=new_selected, selection_order=new_selection_order)


from scipy.sparse import csgraph


def cluster(hypotheses_pool):
    """

    :param hypotheses_pool:
    :param simulator:
    :return:
    """

    """
    COMPUTE ADJACENCY MATRIX
    """

    # TODO is this needed?
    # R, t, K = CAMERA_PARAMETERS
    # simulator.initialize_frame(R, t, K, perspective=True, topview=False)

    # --- initialize scene
    hypotheses = []
    for obj_hypotheses in hypotheses_pool.values():
        hypotheses += [obj_hypotheses]  # new format to old format
    SIMULATOR.objects_to_use = [h.id for hs in hypotheses for h in hs]
    SIMULATOR.reset_objects()

    SIMULATOR.initialize_solution(hypotheses)  # with render_all_hypotheses True this should do the job

    # --- get intersections - compute adjacency matrix
    # step simulation to update contact points
    SIMULATOR.pyb.setTimeStep(0.001)
    SIMULATOR.pyb.setPhysicsEngineParameter(fixedTimeStep=0.001, numSolverIterations=1, numSubSteps=0)
    SIMULATOR.pyb.stepSimulation()

    # get all contact points and compile an adjacency matrix from that
    adjacency = np.eye(len(hypotheses))  # every object is at least adjacent to itself
    obj_ids = list(hypotheses_pool.keys())
    for obj_i, obj_hypotheses in enumerate(hypotheses):
        obj_contacts = []
        for h in obj_hypotheses:
            contact_points = SIMULATOR.pyb.getContactPoints(bodyA=SIMULATOR.models[h.id],
                                                            physicsClientId=SIMULATOR.world)  # TODO instead calling pybullet, add this functionality in simulator

            for contact_point in contact_points:
                other_id = contact_point[2]
                distance = contact_point[8]

                other_str = '?'
                other_i = -1
                for k, v in SIMULATOR.models.items():
                    if v == other_id:
                        other_str = k
                        other_i = obj_ids.index(other_str[:2])  # per model
                # print("%i (%s)" % (other_i, other_str))
                adjacency[obj_i, other_i] = 1

    adjacency = adjacency + adjacency.T  # ensure it is symmetric -> bidirectionality required for clustering
    adjacency[adjacency > 0] = 1

    """
    CLUSTERING
    """

    # bandwidth reduction -> permutation s.t. distance on nonzero entries from the center diagonal is minimized
    r = csgraph.reverse_cuthill_mckee(csgraph.csgraph_from_dense(adjacency), True)

    # via http://raphael.candelier.fr/?blog=Adj2cluster
    # and http://ciprian-zavoianu.blogspot.com/2009/01/project-bandwidth-reduction.html
    # -> results in blocks in the adjacency matrix that correspond with the clusters
    # -> iteratively extend the block while the candidate region contains nonzero elements (i.e. is connected)
    clusters = [[r[0]]]
    for i in range(1, len(r)):
        if np.any(adjacency[clusters[-1], r[i]]):  # ri connected to current cluster? -> add ri to cluster
            clusters[-1].append(r[i])
        else:  # otherwise: start a new cluster with ri
            clusters.append([r[i]])

    # add clustered objects to hypotheses clusters
    hypotheses_clusters = []
    adjacencies = []
    obj_ids = list(hypotheses_pool.keys())
    for cluster in clusters:
        cluster_pool = {}
        cluster_adjacency = {}
        for ci in cluster:
            obj_str = obj_ids[ci]
            obj_hypotheses = hypotheses_pool[obj_str]
            cluster_pool[obj_str] = obj_hypotheses

            # TODO write this python-y
            cluster_adjacency[obj_str] = []
            for ci_ in cluster:
                if ci_ == ci:
                    continue
                if adjacency[ci, ci_] == 1:
                    cluster_adjacency[obj_str].append(obj_ids[ci_])
        hypotheses_clusters.append(cluster_pool)
        adjacencies.append(cluster_adjacency)

    return hypotheses_clusters, adjacencies
