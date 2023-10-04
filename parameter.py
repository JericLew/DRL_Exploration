from datetime import datetime
'''
TODO
- Make it k 20 (change -1,0,1)
- Tune and train
- Optimise graph update #NOTE DONE!, updated sensor range nodes only!
- Optimise graph vertexes
- Tidy DStar implementation
'''

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
SAVE_IMG_GAP = 1 # episode interval for gif saving
SAVE_FREQ = 50 # How often we save model in number of episodes

'''REWARD PARAMETERS'''
FINISHING_REWARD = 10
SAME_POSITION_PUNISHMENT = 0.6
DIST_DENOMINATOR = 48 # 0 to 66, ave 45
FRONTIER_DENOMINATOR = 50 # 0 to 20 , super sparse
REWARD_SCALE_FACTOR = 0.1 # multply to rewards

'''WORKER PARAMETERS'''
NUM_PLANNING_STEP = 16
NUM_ACTION_STEP = 8
MAX_TIMESTEP_PER_EPISODE = NUM_PLANNING_STEP * NUM_ACTION_STEP
MAP_DOWNSIZE_FACTOR = 2 # used to be 2

'''ENV PARAMETERS'''
UNIFORM_POINT_INTERVAL = 30

'''DRIVER PARAMETERS'''
INPUT_DIM = (8,480//MAP_DOWNSIZE_FACTOR,640//MAP_DOWNSIZE_FACTOR)

'''NETWORK PARAMETERS'''
HIDDEN_SIZE = 256

'''TRAINING PARAMETERS'''
LOAD_MODEL = False
SUMMARY_WINDOW = 50
BATCH_SIZE = 128
N_UPDATES_PER_ITERATIONS = 5 # Number of times to update actor/critic per iteration
MINIMUM_BUFFER_SIZE = 500 # 500 for laptop 2000 for desktop
REPLAY_SIZE = 2000 # 2000 for laptop 5000 for desktop

'''PPO HYPERPARAMETERS'''
LR = 2e-5 # Learning rate of actor optimizer
GAMMA = 0.99 # Discount factor to be applied when calculating Rewards-To-Go
CLIP = 0.2 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
MAX_GRAD_NORM = 20
CRITIC_LOSS_COEF = 0.5
ENTROPY_COEF = 0.001