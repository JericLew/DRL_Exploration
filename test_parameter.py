'''LOCAL AND GLOBAL DEVICE'''
USE_GPU = True  # do you want to use GPU to test
NUM_GPU = 1
NUM_META_AGENT = 4 # 4 for laptop 8 for desktop

'''FILE DIRECTORIES AND SAVE FREQUENCIES'''
FOLDER_NAME = "run_2023_09_24_0053"
model_path = f'model/{FOLDER_NAME}'
gifs_path = f'results/{FOLDER_NAME}/gifs'
trajectory_path = f'results/{FOLDER_NAME}/trajectory'
length_path = f'results/{FOLDER_NAME}/length'
SAVE_GIFS = True  # do you want to save GIFs
SAVE_TRAJECTORY = True  # do you want to save per-step metrics
SAVE_LENGTH = True  # do you want to save per-episode metrics

'''WORKER PARAMETERS'''
MAX_TIMESTEP_PER_EPISODE = 120
NUM_PLANNING_STEP = 24
NUM_ACTION_STEP = 5
K_SIZE = 12  # the number of neighboring nodes
MAP_DOWNSIZE_FACTOR = 2

'''NETWORK PARAMETERS'''
INPUT_DIM = (8,240,320)

'''TEST PARAMETERS'''
NUM_TEST = 100
NUM_RUN = 1
