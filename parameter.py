'''LOCAL AND GLOBAL DEVICE'''
USE_GPU = False  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 1
NUM_META_AGENT = 2

'''FILE DIRECTORIES AND SAVE FREQUENCIES'''
FOLDER_NAME = 'ae_clean'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
GLOBAL_SAVE_IMG = True # False to have no image saved at all
SAVE_IMG_GAP = 10 # episode interval for gif saving
SAVE_FREQ = 10 # How often we save in number of iterations

'''REWARD PARAMETERS'''
FINISHING_REWARD = 20 / 10
SAME_POSITION_PUNISHMENT = 1 / 10
DIST_DENOMINATOR = 64 * 10 # 0 to 66, ave 45
FRONTIER_DENOMINATOR = 25 * 10 # 0 to 20 , super sparse

'''NETWORK PARAMETERS'''
INPUT_DIM = (8,240,320)
HIDDEN_SIZE = 256
MAP_DOWNSIZE_FACTOR = 2
K_SIZE = 10  # the number of neighboring nodes

'''PPO HYPERPARAMETERS'''
EPISODE_PER_BATCH = 2 # default 2
MAX_TIMESTEP_PER_EPISODE = 120 # Max number of timesteps per episode default 128
TIMESTEP_PER_BATCH =  EPISODE_PER_BATCH * MAX_TIMESTEP_PER_EPISODE # Number of timesteps to run per batch
N_UPDATES_PER_ITERATIONS = 5 # Number of times to update actor/critic per iteration
LR = 2.5e-5 # Learning rate of actor optimizer
GAMMA = 1 # Discount factor to be applied when calculating Rewards-To-Go
CLIP = 0.2 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
MAX_GRAD_NORM = 0.5
CRITIC_LOSS_COEF = 0.5
ENTROPY_COEF = 0.001
NUM_PLANNING_STEP = 30
NUM_ACTION_STEP = 4