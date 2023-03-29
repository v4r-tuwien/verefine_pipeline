# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

# VEREFINE
HYPOTHESES_PER_OBJECT = 3  # double of PPF to account for flipped hypotheses
REFINEMENT_ITERATIONS = 2
SIMULATION_STEPS = 3
MODE = 3
# MODES = {
#     0: "BASELINE",
#     1: "PIR",
#     2: "SIR",
#     3: "RIR",
#     4: "VFb",
#     5: "VFd"
# }

# PHYSICS SIMULATION
TIME_STEP = 1/60
SOLVER_ITERATIONS = 10
SOLVER_SUB_STEPS = 4

# REFINEMENT
ICP_ITERATIONS = 10  # per VeREFINE iteration
ICP_P_DISTANCE = 0.1

# RENDERING
CLIP_NEAR, CLIP_FAR = 0.01, 5.0  # clipping distances in renderer

# BANDIT
C = 0.1  # exploration rate in UCB
GAMMA = 0.99  # discount factor in D-UCB
