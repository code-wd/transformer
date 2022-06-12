import torch


# init parameters
UNK = 0  # unknown word-id
PAD = 1  # padding word-id
BATCH_SIZE = 64

DEBUG = False    # Debug / Learning Purposes.
# DEBUG = False # Build the model, better with GPU CUDA enabled.

if DEBUG:
    EPOCHS = 2
    LAYERS = 3
    H_NUM = 8
    D_MODEL = 128
    D_FF = 256
    DROPOUT = 0.1
    MAX_LENGTH = 60
    TRAIN_FILE = 'data/nmt/en-cn/train_mini.txt'
    DEV_FILE = 'data/nmt/en-cn/dev_mini.txt'
    SAVE_FILE = 'ckpt/model.pt'
else:
    EPOCHS = 20
    LAYERS = 6
    H_NUM = 8
    D_MODEL = 256
    D_FF = 1024
    DROPOUT = 0.1
    MAX_LENGTH = 60
    TRAIN_FILE = 'data/nmt/en-cn/train.txt'
    DEV_FILE = 'data/nmt/en-cn/dev.txt'
    SAVE_FILE = 'ckpt/large_model.pt'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")