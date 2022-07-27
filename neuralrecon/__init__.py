import os
import sys
import pathlib

# Get the location of __init__.py
cwd = pathlib.Path(os.path.abspath(__file__)).parent

# Creating default configuration
from yacs.config import CfgNode as CN

DF_CF = CN()

DF_CF.MODE = 'train'
DF_CF.DATASET = 'scannet'
DF_CF.BATCH_SIZE = 1
DF_CF.LOADCKPT = ''
DF_CF.LOGDIR = './checkpoints/debug'
DF_CF.RESUME = True
DF_CF.SUMMARY_FREQ = 20
DF_CF.SAVE_FREQ = 1
DF_CF.SEED = 1
DF_CF.SAVE_SCENE_MESH = False
DF_CF.SAVE_INCREMENTAL = False
DF_CF.VIS_INCREMENTAL = False
DF_CF.REDUCE_GPU_MEM = False

DF_CF.LOCAL_RANK = 0
DF_CF.DISTRIBUTED = False

# train
DF_CF.TRAIN = CN()
DF_CF.TRAIN.PATH = ''
DF_CF.TRAIN.EPOCHS = 40
DF_CF.TRAIN.LR = 0.001
DF_CF.TRAIN.LREPOCHS = '12,24,36:2'
DF_CF.TRAIN.WD = 0.0
DF_CF.TRAIN.N_VIEWS = 5
DF_CF.TRAIN.N_WORKERS = 8
DF_CF.TRAIN.RANDOM_ROTATION_3D = True
DF_CF.TRAIN.RANDOM_TRANSLATION_3D = True
DF_CF.TRAIN.PAD_XY_3D = .1
DF_CF.TRAIN.PAD_Z_3D = .025

# test
DF_CF.TEST = CN()
DF_CF.TEST.PATH = ''
DF_CF.TEST.N_VIEWS = 5
DF_CF.TEST.N_WORKERS = 4

# model
DF_CF.MODEL = CN()
DF_CF.MODEL.N_VOX = [128, 224, 192]
DF_CF.MODEL.VOXEL_SIZE = 0.04
DF_CF.MODEL.THRESHOLDS = [0, 0, 0]
DF_CF.MODEL.N_LAYER = 3

DF_CF.MODEL.TRAIN_NUM_SAMPLE = [4096, 16384, 65536]
DF_CF.MODEL.TEST_NUM_SAMPLE = [32768, 131072]

DF_CF.MODEL.LW = [1.0, 0.8, 0.64]

# TODO: images are currently loaded RGB, but the pretrained models expect BGR
DF_CF.MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675]
DF_CF.MODEL.PIXEL_STD = [1., 1., 1.]
DF_CF.MODEL.THRESHOLDS = [0, 0, 0]
DF_CF.MODEL.POS_WEIGHT = 1.0

DF_CF.MODEL.BACKBONE2D = CN()
DF_CF.MODEL.BACKBONE2D.ARC = 'fpn-mnas'

DF_CF.MODEL.SPARSEREG = CN()
DF_CF.MODEL.SPARSEREG.DROPOUT = False

DF_CF.MODEL.FUSION = CN()
DF_CF.MODEL.FUSION.FUSION_ON = False
DF_CF.MODEL.FUSION.HIDDEN_DIM = 64
DF_CF.MODEL.FUSION.AVERAGE = False
DF_CF.MODEL.FUSION.FULL = False

# Subpackages
from . import datasets
from . import ops
from . import models
from .models import NeuralRecon

# Modules
from . import utils
