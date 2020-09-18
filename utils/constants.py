#needs to be modified to my project, but love this idea of applying constant variables that are just passed in.

FFMPEG_EXECUTABLE = "/home/ubuntu/anaconda3/bin/ffmpeg"
SLASH = "/"

# ==========================
#  All Path constants
# ==========================
BASE_PROJECT_PATH = "/home/ubuntu/000_homebase/deepASL/"
BASE_DATA_PATH = BASE_PROJECT_PATH+"dataset/"
BASE_LOG_PATH = BASE_PROJECT_PATH+"saved_models/"

# =====================
#  DATA Relative Path
# =====================
DATA_TRAIN_VIDEOS = "dataset/train"
DATA_VALID_VIDEOS = "dataset/valid"
DATA_TEST_VIDEOS = "dataset/test"
DATA_BG_TRAIN_VIDEO = "dataset/bg_train_data"
DATA_BG_TEST_VIDEO = "dataset/bg_test_data"


# ===========================
# Saved Models Relative Path
# ===========================
MODEL_INIT = "vae/"
MODEL_TRANL = "ssd_mobilenet/"


# ===========================
# PB File Names
# ===========================
INIT_FREEZED_PB_NAME = "init_freezed.pb"
TRANL_FREEZED_PB_NAME = "tran_freezed.pb"
