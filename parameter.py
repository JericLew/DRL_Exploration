from datetime import datetime

'''LOCAL AND GLOBAL DEVICE'''
USE_GPU = False  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 1
NUM_META_AGENT = 4 # 4 for laptop 8 for desktop

'''FILE DIRECTORIES AND SAVE FREQUENCIES'''
now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H%M")
FOLDER_NAME = f'run_{dt_string}'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
GLOBAL_SAVE_IMG = True # False to have no image saved at all
SAVE_IMG_GAP = 50 # episode interval for gif saving
SAVE_FREQ = 32 # How often we save model in number of episodes

'''REWARD PARAMETERS'''
FINISHING_REWARD = 20 / 50
SAME_POSITION_PUNISHMENT = 5 / 50
DIST_DENOMINATOR = 48 * 50 # 0 to 66, ave 45
FRONTIER_DENOMINATOR = 50 * 50 # 0 to 20 , super sparse

'''ENV PARAMETER'''
MAX_TIMESTEP_PER_EPISODE = 120
NUM_PLANNING_STEP = 30
NUM_ACTION_STEP = 4
K_SIZE = 10  # the number of neighboring nodes

'''NETWORK PARAMETERS'''
INPUT_DIM = (8,240,320)
HIDDEN_SIZE = 256
MAP_DOWNSIZE_FACTOR = 2

'''TRAINING PARAMETERS'''
LOAD_MODEL = False
SUMMARY_WINDOW = 20
BATCH_SIZE = 128
N_UPDATES_PER_ITERATIONS = 5 # Number of times to update actor/critic per iteration
MINIMUM_BUFFER_SIZE = 500 # 500 for laptop 2000 for desktop
REPLAY_SIZE = 2500 # 2500 for laptop 10000 for desktop

'''PPO HYPERPARAMETERS'''
LR = 0.5e-5 # Learning rate of actor optimizer
GAMMA = 0.99 # Discount factor to be applied when calculating Rewards-To-Go
CLIP = 0.2 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
MAX_GRAD_NORM = 1
CRITIC_LOSS_COEF = 0.5
ENTROPY_COEF = 0.005