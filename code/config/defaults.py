# from yacs.config import CfgNode as CN
#
# _C = CN()
#
# # SYSTEM部分：系统相关配置
# _C.SYSTEM = CN()  # 创建SYSTEM子节点
# _C.SYSTEM.NUM_CPU = 4  # CPU数量
# _C.SYSTEM.USE_GPU = True  # 是否使用GPU
# _C.SYSTEM.GPU_IDS = [0]                   # which gpus to use for training - list of int, e.g. [0, 1]
# _C.SYSTEM.RNG_SEED = 42  # 随机数种子
#
# # MODEL部分：模型相关配置
# _C.MODEL = CN()  # 创建MODEL子节点
# #_C.MODEL.NET = "DeepLabRecon"                    # available networks from net.models.py file
# _C.MODEL.NET = "DeepLabReconFuseSimpleTrainModified"                    # available networks from net.models.py file
# _C.MODEL.BACKBONE = "resnet"                # choices: ['resnet', 'xception', 'drn', 'mobilenet']
# _C.MODEL.OUT_STRIDE = 16                    # deeplab output stride Deeplab输出步幅
# _C.MODEL.SYNC_BN = None                     # whether to use sync bn (for multi-gpu), None == Auto detect 是否使用同步批量归一化（Sync Batch Normalization）
# _C.MODEL.FREEZE_BN = False
#
# # RECONSTRUCTION部分：重建相关配置
# _C.MODEL.RECONSTRUCTION = CN()
# _C.MODEL.RECONSTRUCTION.LATENT_DIM = 4      # number of channels of latent space
# # 分割模型路径
# #_C.MODEL.RECONSTRUCTION.SEGM_MODEL = "/mnt/datagrid/personal/vojirtom/sod/mcsegm/20210209_112729_903607/checkpoints/checkpoint-best.pth" #resnet 51.6
# # _C.MODEL.RECONSTRUCTION.SEGM_MODEL = "/mnt/datagrid/personal/vojirtom/sod/mcsegm/20210302_234038_642070/checkpoints/checkpoint-best.pth"  #resnet 66.1
# #_C.MODEL.RECONSTRUCTION.SEGM_MODEL = "/mnt/datagrid/personal/vojirtom/sod/mcsegm/20210306_213340_619898/checkpoints/checkpoint-best.pth" #mobilenet 61.2
# #_C.MODEL.RECONSTRUCTION.SEGM_MODEL = "/mnt/datagrid/personal/vojirtom/sod/mcsegm/20210306_214758_686017/checkpoints/checkpoint-best.pth" #xception 50.3
# _C.MODEL.RECONSTRUCTION.SEGM_MODEL = r"D:\SoftwareEngineer\graduation_project\JSRNet\checkpoint-segmentation.pth"
# _C.MODEL.RECONSTRUCTION.SEGM_MODEL_NCLASS = 19  # 19 for cityscapes 分割模型的类别数
# _C.MODEL.RECONSTRUCTION.SKIP_CONN = False  # 是否使用跳跃连接
# _C.MODEL.RECONSTRUCTION.SKIP_CONN_DIM = 32  # 跳跃连接维度
#
# # LOSS部分：损失函数相关配置
# _C.LOSS = CN()
# #_C.LOSS.TYPE = "ReconstructionAnomalyLoss"           # available losses from net.loss.py
# #_C.LOSS.TYPE = "ReconstructionAnomalyLossFuseSimple"           # available losses from net.loss.py
# _C.LOSS.TYPE = "ReconstructionAnomalyLossFuseTrainAux"           # available losses from net.loss.py  使用的损失函数类型
# _C.LOSS.IGNORE_LABEL = 255  # 忽略的标签
# _C.LOSS.SIZE_AVG = True  # 是否对损失进行大小平均
# _C.LOSS.BATCH_AVG = True   # 是否对损失进行批次平均
#
# # EXPERIMENT部分：实验相关配置
# _C.EXPERIMENT= CN()
# _C.EXPERIMENT.NAME = None                   # None == Auto name from date and time 实验的名称将自动根据当前日期和时间来生成。
# # _C.EXPERIMENT.OUT_DIR = "/ssd/temporary/vojirtom/code_temp/sod/training/mcsegm/"  # 输出目录
# _C.EXPERIMENT.OUT_DIR = "/temp"
# _C.EXPERIMENT.EPOCHS = 200                  # number of training epochs
# _C.EXPERIMENT.START_EPOCH = 0
# _C.EXPERIMENT.USE_BALANCED_WEIGHTS = False
# _C.EXPERIMENT.RESUME_CHECKPOINT = None      # path to resume file (stored checkpoint)
# _C.EXPERIMENT.EVAL_INTERVAL = 1             # eval every X epoch
# _C.EXPERIMENT.EVAL_METRIC = "AnomalyEvaluator" # available evaluation metrics from utils.metrics.py file  # 定义评估指标
#
# _C.INPUT = CN()
# _C.INPUT.BASE_SIZE = 512
# _C.INPUT.CROP_SIZE = 512
# _C.INPUT.NORM_MEAN = [0.485, 0.456, 0.406]  # mean for the input image to the net (image -> (0, 1) -> mean/std)
# _C.INPUT.NORM_STD = [0.229, 0.224, 0.225]   # std for the input image to the net (image -> (0, 1) -> mean/std)
# _C.INPUT.BATCH_SIZE_TRAIN = 2           # None = Auto set based on training dataset
# _C.INPUT.BATCH_SIZE_TEST = 2             # None = Auto set based on training batch size
#
# _C.AUG = CN()
# _C.AUG.RANDOM_CROP_PROB = 0.5               # prob that random polygon (anomaly) will be cut from image vs. random noise
# _C.AUG.SCALE_MIN = 0.5
# _C.AUG.SCALE_MAX = 2.0
# _C.AUG.COLOR_AUG = 0.25
#
# _C.OPTIMIZER = CN()
# _C.OPTIMIZER.LR = 0.001
# _C.OPTIMIZER.LR_SCHEDULER = "poly"          # choices: ['poly', 'step', 'cos']  # 定义学习率调度器类型
# _C.OPTIMIZER.MOMENTUM = 0.9
# _C.OPTIMIZER.WEIGHT_DECAY = 5e-4
# _C.OPTIMIZER.NESTEROV = False
#
# _C.DATASET = CN()
# _C.DATASET.TRAIN = "LaF"      # choices: ['cityscapes'],
# _C.DATASET.VAL = "LaF"                      # choices: ['cityscapes'],
# _C.DATASET.TEST = "LaF"                     # choices: ['LaF'],
# _C.DATASET.FT = False                       # flag if we are finetuning
#
# _C.DATASET.PATH = CN()
# _C.DATASET.PATH.CITYSCAPES=r'D:\SoftwareEngineer\graduation_project\datasets\cityscapes'
# _C.DATASET.PATH.LAF=r'D:\SoftwareEngineer\graduation_project\datasets\lost_and_found'
#
#
#
# def get_cfg_defaults():
#     """Get a yacs CfgNode object with default values for my_project."""
#     return _C.clone()
#
from yacs.config import CfgNode as CN

_C = CN()

# SYSTEM部分：系统相关配置
_C.SYSTEM = CN()  # 创建SYSTEM子节点
_C.SYSTEM.NUM_CPU = 4  # CPU数量
_C.SYSTEM.USE_GPU = True  # 是否使用GPU
_C.SYSTEM.GPU_IDS = [0]                   # which gpus to use for training - list of int, e.g. [0, 1]
_C.SYSTEM.RNG_SEED = 42  # 随机数种子

# MODEL部分：模型相关配置
_C.MODEL = CN()  # 创建MODEL子节点
#_C.MODEL.NET = "DeepLabRecon"                    # available networks from net.models.py file
_C.MODEL.NET = "DeepLabReconFuseSimpleTrainModified"                    # available networks from net.models.py file
_C.MODEL.BACKBONE = "resnet"                # choices: ['resnet', 'xception', 'drn', 'mobilenet']
_C.MODEL.OUT_STRIDE = 16                    # deeplab output stride Deeplab输出步幅
_C.MODEL.SYNC_BN = None                     # whether to use sync bn (for multi-gpu), None == Auto detect 是否使用同步批量归一化（Sync Batch Normalization）
_C.MODEL.FREEZE_BN = False

# RECONSTRUCTION部分：重建相关配置
_C.MODEL.RECONSTRUCTION = CN()
_C.MODEL.RECONSTRUCTION.LATENT_DIM = 4      # number of channels of latent space
# 分割模型路径
# _C.MODEL.RECONSTRUCTION.SEGM_MODEL = "/mnt/datagrid/personal/vojirtom/sod/mcsegm/20210209_112729_903607/checkpoints/checkpoint-best.pth" #resnet 51.6
# _C.MODEL.RECONSTRUCTION.SEGM_MODEL = "/mnt/datagrid/personal/vojirtom/sod/mcsegm/20210302_234038_642070/checkpoints/checkpoint-best.pth"  #resnet 66.1
#_C.MODEL.RECONSTRUCTION.SEGM_MODEL = "/mnt/datagrid/personal/vojirtom/sod/mcsegm/20210306_213340_619898/checkpoints/checkpoint-best.pth" #mobilenet 61.2
#_C.MODEL.RECONSTRUCTION.SEGM_MODEL = "/mnt/datagrid/personal/vojirtom/sod/mcsegm/20210306_214758_686017/checkpoints/checkpoint-best.pth" #xception 50.3
_C.MODEL.RECONSTRUCTION.SEGM_MODEL = r"D:\SoftwareEngineer\graduation_project\JSRNet\checkpoint-segmentation.pth"
_C.MODEL.RECONSTRUCTION.SEGM_MODEL_NCLASS = 19  # 19 for cityscapes 分割模型的类别数
_C.MODEL.RECONSTRUCTION.SKIP_CONN = False  # 是否使用跳跃连接
_C.MODEL.RECONSTRUCTION.SKIP_CONN_DIM = 32  # 跳跃连接维度
_C.MODEL.PRETRAINED = r"D:\SoftwareEngineer\graduation_project\JSRNet\records\experiments\masked_experiment_L_L_896\checkpoints\checkpoint-best.pth"

# LOSS部分：损失函数相关配置
_C.LOSS = CN()
#_C.LOSS.TYPE = "ReconstructionAnomalyLoss"           # available losses from net.loss.py
#_C.LOSS.TYPE = "ReconstructionAnomalyLossFuseSimple"           # available losses from net.loss.py
_C.LOSS.TYPE = "ReconstructionAnomalyLossFuseTrainAux"           # available losses from net.loss.py  使用的损失函数类型
_C.LOSS.IGNORE_LABEL = 255  # 忽略的标签
_C.LOSS.SIZE_AVG = True  # 是否对损失进行大小平均
_C.LOSS.BATCH_AVG = True   # 是否对损失进行批次平均

# EXPERIMENT部分：实验相关配置
_C.EXPERIMENT= CN()
_C.EXPERIMENT.NAME = None                   # None == Auto name from date and time 实验的名称将自动根据当前日期和时间来生成。
# _C.EXPERIMENT.OUT_DIR = "/ssd/temporary/vojirtom/code_temp/sod/training/mcsegm/"  # 输出目录
_C.EXPERIMENT.OUT_DIR = r"D:\SoftwareEngineer\graduation_project\JSRNet\temp"
_C.EXPERIMENT.EPOCHS = 200                  # number of training epochs
_C.EXPERIMENT.START_EPOCH = 0
_C.EXPERIMENT.USE_BALANCED_WEIGHTS = False
_C.EXPERIMENT.RESUME_CHECKPOINT = None      # path to resume file (stored checkpoint)
_C.EXPERIMENT.EVAL_INTERVAL = 1             # eval every X epoch
_C.EXPERIMENT.TEST_INTERVAL = 5             # test every X epoch
_C.EXPERIMENT.EVAL_METRIC = "AnomalyEvaluator" # available evaluation metrics from utils.metrics.py file  # 定义评估指标

_C.INPUT = CN()
_C.INPUT.BASE_SIZE = 896
_C.INPUT.CROP_SIZE = 896
_C.INPUT.NORM_MEAN = [0.485, 0.456, 0.406]  # mean for the input image to the net (image -> (0, 1) -> mean/std)
_C.INPUT.NORM_STD = [0.229, 0.224, 0.225]   # std for the input image to the net (image -> (0, 1) -> mean/std)
_C.INPUT.BATCH_SIZE_TRAIN = 2           # None = Auto set based on training dataset
_C.INPUT.BATCH_SIZE_TEST = 2             # None = Auto set based on training batch size

_C.AUG = CN()
_C.AUG.RANDOM_CROP_PROB = 0.5               # prob that random polygon (anomaly) will be cut from image vs. random noise
_C.AUG.SCALE_MIN = 0.5
_C.AUG.SCALE_MAX = 2.0
_C.AUG.COLOR_AUG = 0.25

_C.OPTIMIZER = CN()
_C.OPTIMIZER.LR = 0.001
_C.OPTIMIZER.LR_SCHEDULER = "poly"          # choices: ['poly', 'step', 'cos']  # 定义学习率调度器类型
_C.OPTIMIZER.MOMENTUM = 0.9
_C.OPTIMIZER.WEIGHT_DECAY = 5e-4
_C.OPTIMIZER.NESTEROV = False

_C.DATASET = CN()
_C.DATASET.TRAIN = "LaF"      # choices: ['cityscapes'],
_C.DATASET.VAL = "OT"                      # choices: ['cityscapes'],
_C.DATASET.TEST = "OT"                     # choices: ['LaF'],
_C.DATASET.FT = False                       # flag if we are finetuning

_C.DATASET.PATH = CN()
_C.DATASET.PATH.CITYSCAPES=r'D:\SoftwareEngineer\graduation_project\datasets\cityscapes'
_C.DATASET.PATH.LAF=r'D:\SoftwareEngineer\graduation_project\datasets\lost_and_found'
_C.DATASET.PATH.OBSTACLE_TRACK = r'D:\SoftwareEngineer\graduation_project\datasets\dataset_ObstacleTrack'
_C.DATASET.PATH.ROADANOMALY21 = r'D:\SoftwareEngineer\graduation_project\datasets\dataset_AnomalyTrack'
_C.DATASET.PATH.ROADANOMALY = r'D:\SoftwareEngineer\graduation_project\datasets\RoadAnomaly'

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()
