USE_GPU = False  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = False  # do you want to train the network using GPUs

FOLDER_NAME = 'ae_clean'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
SAVE_IMG_GAP = 10
SAVE_FREQ = 10 # How often we save in number of iterations


# Algorithm hyperparameters
INPUT_DIM = (8,240,320)
HIDDEN_SIZE = 256
K_SIZE = 20  # the number of neighboring nodes
EPISODE_PER_BATCH = 2 # default 2
MAX_TIMESTEP_PER_EPISODE = 128 # Max number of timesteps per episode default 128
TIMESTEP_PER_BATCH =  EPISODE_PER_BATCH * MAX_TIMESTEP_PER_EPISODE # Number of timesteps to run per batch
N_UPDATES_PER_ITERATIONS = 5 # Number of times to update actor/critic per iteration
LR = 1e-5 # Learning rate of actor optimizer
GAMMA = 0.95 # Discount factor to be applied when calculating Rewards-To-Go
CLIP = 0.2 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
