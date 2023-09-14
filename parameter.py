'''LOCAL AND GLOBAL DEVICE'''
USE_GPU = False  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs

'''FILE DIRECTORIES AND SAVE FREQUENCIES'''
FOLDER_NAME = 'ae_clean'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
SAVE_IMG_GAP = 4 # episode interval for gif saving
GLOBAL_SAVE_IMG = True # False to have no image saved at all
SAVE_FREQ = 10 # How often we save in number of iterations

'''REWARD PARAMETERS'''
FINISHING_REWARD = 20 / 10
SAME_POSITION_PUNISHMENT = 5 / 10 
DIST_DENOMINATOR = 64 * 10 # 0 to 66, ave 45
FRONTIER_DENOMINATOR = 25 * 10 # 0 to 20 , super sparse

'''NETWORK PARAMETERS'''
INPUT_DIM = (8,240,320)
HIDDEN_SIZE = 256
MAP_DOWNSIZE_FACTOR = 2
K_SIZE = 20  # the number of neighboring nodes

'''PPO HYPERPARAMETERS'''
EPISODE_PER_BATCH = 1 # default 2
MAX_TIMESTEP_PER_EPISODE = 128 # Max number of timesteps per episode default 128
TIMESTEP_PER_BATCH =  EPISODE_PER_BATCH * MAX_TIMESTEP_PER_EPISODE # Number of timesteps to run per batch
N_UPDATES_PER_ITERATIONS = 8 # Number of times to update actor/critic per iteration
LR = 1e-5 # Learning rate of actor optimizer
GAMMA = 1 # Discount factor to be applied when calculating Rewards-To-Go
CLIP = 0.2 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
CRITIC_LOSS_COEF = 0.5
ENTROPY_COEF = 0.001