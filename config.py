# ROOT = '/usr/not-backed-up/scnb/data/LAX-fullcollection'
ROOT = '/app/data/data/LAX-fullcollection'

# TRAIN_ROOT = '/usr/not-backed-up/scnb/data/LAX-fullcollection'
TRAIN_ROOT = '/app/data/data/LAX-fullcollection'

TEST_ROOT = "/usr/not-backed-up/scnb/data/LAX-uncropped"

# MODEL_DIR = "/usr/not-backed-up/scnb/motion_model/model_full_64by64_newlossc6_27042021.plt"
MODEL_DIR = "/app/data/models/model_full_64by64_avgdt_nobn_12052021.plt"

# CURRENT_MODEL_DIR = "/usr/not-backed-up/scnb/motion_model/current_model_full_64by64_newlossc6_27042021.plt"
CURRENT_MODEL_DIR = "/app/data/models/model_current_64by64_avgdt_nobn_12052021.plt"

# SAVE_FIG_ROOT = '/app/data/test_figs'
SAVE_FIG_ROOT = '/usr/not-backed-up/scnb/test_figs'
CUDA_DEVICE = 'cuda:7'
TEST_CUDA_DEVICE = 'cuda:7'
NUM_OF_FRAMES = 5
TRAIN_RESIZE = (128,128)
RESIZE_WIDTH = 128
N_Z = 64
N_D = RESIZE_WIDTH*RESIZE_WIDTH
HIDDEN_SIZE = 16*16*64
H_DIM = 64
PHIX_DIM = 64
N_I = RESIZE_WIDTH*RESIZE_WIDTH

BATCH_SIZE = 10
SHUFFLE = True
NUM_WORKERS = 64

NUM_OF_SAMPLES = 5

