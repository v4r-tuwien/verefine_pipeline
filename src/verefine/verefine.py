# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

import numpy as np
import util.rotation as r  # TODO available in bop_toolkit?


# ===================
# ===== CONFIG ======  TODO set via parameters in experiment
# ===================
USE_BASELINE_HYPOTHESES = False  # whether to use precomputed baseline hypotheses pool (True) or recompute (False)
USE_POSECNN = False  # load posecnn's results as pose estimator -- note: only one hypothesis per object
USE_COLLISION_CLUSTERING = True  # else rendering-based version
USE_ICP = False  # else use DF as refiner
FIX_OTHERS = True  # whether to fix supporting objects in their best hypothesis (True) or simulate them as well (False)
EXPAND_ADJACENT = True  # whether to only expand adjacent objects (True) or based on <2cm displacement (False)
# POLICY = UCB
C = 0.1 #

ALL_OBJECTS = True
HYPOTHESES_PER_OBJECT = 2
ITERATIONS = 2
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


def fit(observation, rendered):
    depth_obs = observation['depth']
    depth_ren = rendered[1] * 1000  # in mm TODO or do this in renderer?

    mask = np.logical_and(depth_ren > 0, depth_obs > 0)
    if np.count_nonzero(mask) == 0:  # no valid depth values
        return 0

    mask = np.logical_and(mask, depth_ren - depth_obs < TAU_VIS)  # only visible -- ren at most [TAU_VIS] behind obs
    dist = np.abs(depth_obs[mask] - depth_ren[mask])
    delta = np.mean(np.min(np.vstack((dist / TAU, np.ones(dist.shape[0]))), axis=0))
    visibility_ratio = mask.sum() / (depth_ren > 0).sum()  # visible / rendered count

    fit = visibility_ratio * (1 - delta)
    if np.isnan(fit):  # invisible object
        return 0

    return fit


class Hypothesis:
    """
    A hypothesis giving a candidate object (model) and a candidate transformation.
    - "We assume that the object is present in the scene, under the given transformation."
    """

    def __init__(self, model, transformation, roi, mask, embedding, cloud, instance_id=0, confidence=1.0, grasps=[]):
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
        self.grasp_transformations = grasps

    def duplicate(self):
        """
        TODO
        :return:
        """
        return Hypothesis(model=self.model, transformation=self.transformation.copy(), roi=self.roi, mask=self.mask,
                          embedding=self.embedding,  # TODO compute embedding ad-hoc without estimation (for loading)
                          cloud=self.cloud,  # TODO for reproducibility -- the samples that were selected
                          instance_id=self.instance_id, confidence=self.confidence, grasps=self.grasp_transformations)

    def render(self, observation, mode):
        """
        TODO
        :param observation:
        :param mode: color, depth, depth+seg or color+depth+seg
        :return: color in TODO, depth in meters, instance segmentation TODO
        """
        obj_id = RENDERER.dataset.objlist.index(int(self.id[:2]))  # TODO do this conversion in renderer

        assert mode in ['color', 'depth', 'depth+seg', 'color+depth+seg']

        # TODO just pass a list of hypotheses alternatively
        rendered = RENDERER.render([obj_id], [self.transformation],
                                   observation['extrinsics'], observation['intrinsics'],
                                   mode=mode)#, bbox=self.roi)

        return rendered

    def fit(self, observation):
        """
        TODO mention PoseRBPF
        :param observation:
        :param rendered:
        :return:
        """

        rendered = self.render(observation, mode='depth')

        return fit(observation, rendered)


# ==================
# ===== PHYSIR =====
# ==================

from verefine.refiner_interface import Refiner


class PhysIR(Refiner):

    """
    TODO
    """

    def __init__(self, refiner):
        Refiner.__init__(self)

        self.refiner = refiner
        self.fixed = []

        # TODO where to move this config stuff?
        self.PHYSICS_STEPS = 20
        self.REFINEMENTS_PER_ITERATION = 1

    def refine(self, rgb, depth, intrinsics, obj_roi, obj_mask, obj_id,
               estimate, iterations=1):
        """
        TODO
        :param hypothesis:
        :param fixed:
        :return:
        """

        SIMULATOR.objects_to_use = [hypothesis.id] + [fix[0].id for fix in self.fixed]
        SIMULATOR.reset_objects()

        physics_hypotheses = []
        for ite in range(0, iterations):
            # === PHYSICS
            # on LM: 5ms, 100 * (iteration+1) steps, just use rotation
            solution = [[hypothesis]] + self.fixed
            SIMULATOR.initialize_solution(solution)

            # if config.FIX_OTHERS:
            #     for fix in self.fixed:
            #         self.simulator.fix_hypothesis(fix[0])

            step_per_iter = self.PHYSICS_STEPS  # 20 on YCB, 100 on LM -- single hyp: *(iteration+1); multi hyp: constant
            steps = step_per_iter  # TODO step_per_iter * (iteration+1)
            T_phys = SIMULATOR.simulate_no_render(hypothesis.id, delta=0.005, steps=steps)

            # if config.FIX_OTHERS:
            #     self.simulator.unfix()

            # just use the rotation -- both are in camera system, so do not need to change the offset
            hypothesis.transformation[:3, :3] = T_phys[:3, :3]  # only R

            physics_hypotheses.append(hypothesis.duplicate())

            # === DF
            self.refiner.refine(rgb, depth, intrinsics, obj_roi, obj_mask, obj_id,
                                estimate, iterations=self.REFINEMENTS_PER_ITERATION)
        return physics_hypotheses


# ================================================================
# ===== BUDGET ALLOCATION BANDIT (Object-level Verification) =====
# ================================================================

class BudgetAllocationBandit:

    """
    TODO
    """

    def __init__(self, pir, observation, hypotheses):
        self.pir = pir
        self.observation = observation

        self.pool = [[h] for h in hypotheses + [None] * len(hypotheses)]  # for the phys hypotheses
        self.max_iter = MAX_REFINEMENTS_PER_HYPOTHESIS + len(hypotheses)  # for initial play that is no refinement

        # ---
        # INIT
        arms = len(hypotheses)
        self.fits = np.zeros((arms * 2, self.max_iter + 1))
        for hi, h in enumerate(hypotheses):
            self.fits[hi] = h.fit(observation)

        self.plays = [1] * arms
        self.rewards = [fit for fit in self.fits[:, 0]]  # init reward with fit of initial hypothesis

    # TODO caller has to handle the fixing of the environment
    def refine(self, fixed=[]):
        iteration = np.sum(self.plays)
        if iteration < self.max_iter:
            # SELECT
            ucb_scores = [r + np.sqrt(0.001) * np.sqrt(np.log(iteration) / n) for r, n in zip(self.rewards, self.plays)]
            hi = np.argmax(ucb_scores)

            h = self.pool[hi][-1]  # most recent refinement of this hypothesis

            # EXPAND
            child = h.duplicate()

            # ROLLOUT
            self.pir.fixed = fixed
            physics_hypotheses = self.pir.refine(child, 1)
            physics_child = physics_hypotheses[0]
            self.pir.fixed = []

            # REWARD
            reward = child.fit(self.observation)
            reward_phys = physics_child.fit(self.observation)

            # BACKPROPAGATE
            self.fits[hi][self.plays[hi]] = reward
            self.fits[hi + len(self.plays)][self.plays[hi]] = reward_phys
            self.pool[hi].append(child)  # hypothesis after physics + refine
            self.pool[hi + len(self.plays)].append(physics_child)  # hypothesis after physics

            self.rewards[hi] = (self.rewards[hi] * float(self.plays[hi]) + reward) / (
                        self.plays[hi] + 1.0)  # running mean

            self.plays[hi] += 1

    def refine_max(self, fixed=[]):
        for iteration in range(self.max_iter - np.sum(self.plays)):  # reduce by already played refinements
            self.refine(fixed)

    def get_best(self):
        # select best fit and add it to final solution
        best_hi, best_ri = np.unravel_index(self.fits.argmax(), self.fits.shape)

        return self.pool[best_hi][best_ri], self.plays[
            best_hi % len(self.plays)] - 1, self.fits.max()  # do not count initial render


# =========================================
# ===== MCTS Scene-level Verification =====
# =========================================

def verefine_solution(hypotheses_pool):
    """
    :param hypotheses_pool: dict with (obj_id, list of Hypothesis)
    :return:
    """

    """
    CLUSTER
    """
    if ALL_OBJECTS:
        hypotheses_pools, adjacencies = cluster(hypotheses_pool)
        # print("  %i cluster(s)" % len(hypotheses_pools))

        # TODO debug -- all in one cluster
        # hypotheses_pools = [hypotheses_pool]
    else:
        # TODO debug -- this should yield same result as verfine_individual
        hypotheses_pools = []
        adjacencies = []
        for obj_str, obj_hypotheses in hypotheses_pool.items():
            hypotheses_pools.append({obj_str: obj_hypotheses})
            adjacencies.append({})

    """
    VEReFINE
    """
    final_hypotheses = []
    final_reward = 0
    for hypotheses_pool, adjacency in zip(hypotheses_pools, adjacencies):
        initial_solution = VerificationSolution(candidates=(hypotheses_pool, adjacency))
        root = SceneVerificationNode(initial_solution)

        all_obj_strs = list(hypotheses_pool.keys())
        nobjects = len(hypotheses_pool.keys())
        max_iterations = int(min(MAX_MCTS_ITERATIONS_PER_OBJECT * nobjects * BUDGET_SCALE, 300))
        if DEBUG_PLOT:
            print("   - #obj_initial = %i" % nobjects)
            print("   - verifine for %i iterations..." % (max_iterations))
        for iteration in range(max_iterations):
            # --- SELECTION - apply selection policy as long as we have the statistics (i.e., fully expanded)
            selected_node = root.select()  #

            # --- EXPANSION - apply expansion policy to get new node
            new_node = selected_node.expand(use_heuristic=USE_HEURISTIC_EXPANSION)

            # --- ROLLOUT - nested bandit to generate rollout policy - compute scene reward for tree
            reward = new_node.rollout()

            # --- BACKPROPAGATION - update statistics of nodes that we descended through
            new_node.backpropagate(reward)
        if DEBUG_PLOT:
            print("%s, %0.3f" % tuple(root.get_best()))

        # === B) get best fit per object
        best_fit_per_object = {}
        best_h_per_object = {}
        for obj_str in list(hypotheses_pool.keys()):
            best_fit_per_object[obj_str] = -1
            best_h_per_object[obj_str] = None

        # gather best hypotheses (per object) in tree
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

        best_per_object(root)

        # to old format + add missing objects
        cluster_hypotheses = []
        for obj_str, h_best in best_h_per_object.items():
            # # TODO missing object; refine it without considering other objects
            if h_best is None:
                refiner = BudgetAllocationBandit(REFINER, OBSERVATION, hypotheses_pool[obj_str])
                for iteration in range(ITERATIONS):
                    refiner.refine()
                h_best = refiner.get_best()[0]

            if h_best is not None:
                cluster_hypotheses.append([h_best])

        # ===
        if DEBUG_PLOT:
            print("   - #obj_verifine = %i\n" % len(cluster_hypotheses))

        final_hypotheses += cluster_hypotheses
        final_reward += reward  # TODO actually would have to render scene with all clusters and compute overall reward
    return final_hypotheses, final_reward


class SceneVerificationNode:

    """
    Node of search tree
    """

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
        for current in descent_path:

            if current.refiner is None:
                assert current.parent is None  # this should only be the case for the root -- TODO unless we add a refiner for pose hypotheses
                continue  # TODO then we remove this continue

            # === REFINER
            if REFINE_AT_ONCE:
                current.refiner.refine_max()#TODO fixed=selected_hypotheses) is this already covered by adding fix_hypothesis and unfix below?
            else:  # only one iteration
                current.refiner.refine()#fixed=selected_hypotheses)

            # === update fixed objects
            # get best hypothesis up till now
            best_hypothesis = current.refiner.get_best()[0]

            # # fix this object (with best hypothesis) in simulation for further refinements in descent path
            SIMULATOR.fix_hypothesis(best_hypothesis)

            # store for final solution
            selected_hypotheses.append([best_hypothesis])
        # unfix hypotheses
        SIMULATOR.unfix()

        # 2) randomly select an hypothesis for unplaced objects, simulate the scene with the known ones fixed
        # -- get list of all objects in cluster
        root = self
        while root.parent is not None:
            root = root.parent
        all_objects = root.solution.initial_candidates[0]
        # --- randomly select one hypothesis of the missing objects
        selected_names = [obj_hs[0].model for obj_hs in selected_hypotheses]
        missing_hypotheses = [[np.random.choice(obj_hyps)] for obj_name, obj_hyps in all_objects.items() if
                              obj_name not in selected_names]
        # --- add to already fixed hypotheses
        scene_hypotheses = selected_hypotheses + missing_hypotheses

        # 3) render resulting scene and compute fitness
        obj_ids = [RENDERER.dataset.objlist.index(int(obj_hs[0].id[:2])) for obj_hs in scene_hypotheses]  # TODO do this conversion in renderer
        obj_trafos = [obj_hs[0].transformation for obj_hs in scene_hypotheses]
        # TODO just pass a list of hypotheses alternatively
        rendered = RENDERER.render(obj_ids, obj_trafos,
                                   OBSERVATION['extrinsics'], OBSERVATION['intrinsics'],
                                   mode='depth')
        scene_reward = fit(OBSERVATION, rendered)

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

        # print("-> selected %s" % self.selection_order[-1])
        adjacency = list(self.candidates[0].keys())  # TODO could actually compute this for floor as well - then we even save the drop test
        if EXPAND_ADJACENT:
            if len(self.selection_order) > 0 and len(self.candidates[1].keys()) > 0:

                adjacency = []
                for selected_obj in self.selection_order:
                    # check if they are adjacent to already placed objects
                    adjacency += self.candidates[1][selected_obj]

                # print("-> selected %s" % self.selection_order[-1])
                # print("---> candidates are %s" % adjacency)

        # get candidates
        candidates = []
        for hi, (obj_id, obj_hs) in enumerate(self.candidates[0].items()):
            if obj_id in adjacency or adjacency is None:
                h = obj_hs[0]  # TODO only take max-conf one? or current best? or use all and select most stable?
                candidates.append(h)

        if len(candidates) == 0:
            return

        # fix current base in its best poses
        fixed = []
        fixed_ids = []
        for obj_id, refiner in self.selected.items():
            h = refiner.get_best()[0]
            fixed_ids.append(h.id)
            fixed.append(h)
            self.init_for_simulation(h, mass=0, collision_group=1, collision_mask=1)#base_mask)

        displacements = []
        for hi, h in enumerate(candidates):
            if h is not None:
                SIMULATOR.objects_to_use = [h.id] + fixed_ids
                SIMULATOR.reset_objects()

                # TODO why does this not work with all at once in different collision groups?
                self.init_for_simulation(h, mass=1, collision_group=2 + hi, collision_mask=1)
                ds, Ts = self.get_displacements([h], delta=0.01, steps=50)
                displacements.append(ds[0])

        # reject candidates with a too high displacement
        th = 0.02
        inliers = np.array(displacements) <= th  # TODO also filter-out obvious FPs (rendering fit < 0.1)

        # remove outliers
        if inliers.sum() == 0:  # s.t. we do not throw away everything
            inliers[np.argmin(displacements)] = True
        for candidate, is_inlier in zip(candidates, inliers):
            if not is_inlier:
                self.candidates[0].pop(candidate.id)

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
        orientation = r.matrix_to_quaternion(R)  # quaternion of form [w, x, y, z]
        orientation = orientation[1:] + [orientation[0]]  # pybullet expects [x, y, z, w]
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
            orientation = [orientation[-1]] + orientation[:-1]  # pybullet returns [x, y, z, w]

            T_after = np.matrix(np.eye(4))
            T_after[0:3, 0:3] = r.quaternion_to_matrix(orientation)
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
        refinement_solution = BudgetAllocationBandit(REFINER, OBSERVATION, candidate_hypotheses)
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
        if np.any(adjacency[
                      clusters[-1], r[
                          i]]):  # ri connected to current cluster? -> add ri to cluster
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
