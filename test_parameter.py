'''LOCAL AND GLOBAL DEVICE'''
USE_GPU = False  # do you want to use GPU to test
NUM_GPU = 1
NUM_META_AGENT = 4 # 4 for laptop 8 for desktop

'''FILE DIRECTORIES AND SAVE FREQUENCIES'''
FOLDER_NAME = ""
model_path = f'model/{FOLDER_NAME}'
gifs_path = f'results/{FOLDER_NAME}/gifs'
trajectory_path = f'results/{FOLDER_NAME}/trajectory'
length_path = f'results/{FOLDER_NAME}/length'
SAVE_GIFS = True  # do you want to save GIFs
SAVE_TRAJECTORY = True  # do you want to save per-step metrics
SAVE_LENGTH = True  # do you want to save per-episode metrics

'''WORKER PARAMETERS'''
NUM_PLANNING_STEP = 32
NUM_ACTION_STEP = 4
MAX_TIMESTEP_PER_EPISODE = NUM_PLANNING_STEP * NUM_ACTION_STEP
MAP_DOWNSIZE_FACTOR = 2

'''NETWORK PARAMETERS'''
INPUT_DIM = (8,480//MAP_DOWNSIZE_FACTOR,640//MAP_DOWNSIZE_FACTOR)

'''TEST PARAMETERS'''
NUM_TEST = 100
NUM_RUN = 1
