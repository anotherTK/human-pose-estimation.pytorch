
import os
import os.path as osp
from yacs.config import CfgNode as CN

_C = CN()
_C.OUTPUT_DIR = "./work_dirs"
_C.RUN_EFFICIENT = False
_C.DTYPE = "float32"
_C.AMP_VERBOSE = False

_C.MODEL = CN()
_C.MODEL.ARCH = "MSPN"
_C.MODEL.BACKBONE = "Res-50"
_C.MODEL.UPSAMPLE_CHANNEL_NUM = 256
_C.MODEL.STAGE_NUM = 2
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHT = ""

_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 5e-3
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.CHECKPOINT_PERIOD = 2400
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)
_C.SOLVER.IMS_PER_GPU = 32
_C.SOLVER.MAX_ITER = 96000
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.OPTIMIZER = 'Adam'
_C.SOLVER.SCHEDULER = 'linear'
_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_ITERS = 2400
_C.SOLVER.WARMUP_METHOD = 'linear'
_C.SOLVER.WEIGHT_DECAY = 1e-5
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.LOSS = CN()
_C.LOSS.OHKM = True
_C.LOSS.TOPK = 8
_C.LOSS.COARSE_TO_FINE = True

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.ASPECT_RATIO_GROUPING = False
_C.DATALOADER.SIZE_DIVISIBILITY = 0

_C.DATA = CN()
_C.DATA.INPUT_SHAPE = (256, 192)  # height, width of human body
_C.DATA.OUTPUT_SHAPE = (64, 48)  # output heatmap size
_C.DATA.PIXEL_STD = 200
_C.DATA.COLOR_RGB = False
_C.DATA.NORMALIZE = True
_C.DATA.MEANS = [0.406, 0.456, 0.485]  # bgr
_C.DATA.STDS = [0.225, 0.224, 0.229]

_C.DATA.DATASET_NAME = "COCO"
_C.DATA.DATASET_ROOT = "datasets/coco"
_C.DATA.BASIC_EXTENTION = 0.05
_C.DATA.RANDOM_EXTENTION = True
_C.DATA.X_EXTENTION = 0.6
_C.DATA.Y_EXTENTION = 0.8
_C.DATA.SCALE_FACTOR_LOW = -0.25
_C.DATA.SCALE_FACTOR_HIGH = 0.25
_C.DATA.SCALE_SHRINK_RATIO = 0.8
_C.DATA.ROTATION_FACTOR = 45
_C.DATA.PROB_ROTATION = 0.5
_C.DATA.PROB_FLIP = 0.5
_C.DATA.NUM_KEYPOINTS_HALF_BODY = 3
_C.DATA.PROB_HALF_BODY = 0.3
_C.DATA.X_EXTENTION_HALF_BODY = 0.6
_C.DATA.Y_EXTENTION_HALF_BODY = 0.8
_C.DATA.ADD_MORE_AUG = False
_C.DATA.GAUSSIAN_KERNELS = [(15, 15), (11, 11), (9, 9), (7, 7), (5, 5)]

# image transform
_C.DATA.BRIGHTNESS = 0.0
_C.DATA.CONTRAST = 0.0
_C.DATA.SATURATION = 0.0
_C.DATA.HUE = 0.0

_C.KEYPOINT = CN()
# number of joints
_C.KEYPOINT.NUM = 17
# mirror correspondence between joints
_C.KEYPOINT.FLIP_PAIRS = [[1, 2], [3, 4], [5, 6],
                          [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
# keypoints belong to upper body
_C.KEYPOINT.UPPER_BODY_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
_C.KEYPOINT.LOWER_BODY_IDS = [11, 12, 13, 14, 15, 16]
_C.KEYPOINT.LOAD_MIN_NUM = 1

_C.TEST = CN()
_C.TEST.FLIP = True
_C.TEST.X_EXTENTION = 0.01 * 9.0
_C.TEST.Y_EXTENTION = 0.015 * 9.0
_C.TEST.SHIFT_RATIOS = [0.25]
_C.TEST.GAUSSIAN_KERNEL = 5
_C.TEST.IMS_PER_GPU = 32
