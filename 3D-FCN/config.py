'''
    3D-FCN config file
'''
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.TAG = "FCN-0009"                    # TAG (It has to be FCN-xxxx)
__C.MODEL = __C.TAG.split('-')[0]       
__C.LR = 0.001                          # Learning Rate           
__C.ALPHA = 1
__C.BETA = 5
__C.ETA = 10
__C.GAMMA = 5
__C.BATCH_NUM = 1
__C.MAX_EPOCH = 60
# DATASET
__C.DATA_DIR = '/usr/app/KittiData/'    # Change this line to your KITTI Dir
__C.DATA_FORMAT='bin'
__C.LABEL_TYPE='txt'
__C.DATA_SETS_TYPE = 'kitti'
__C.CLS = 'Car'                         # Change this line to 'Car' or 'Pedestrain' or 'Cyclist' for different type of prediction
# SHAPE
if __C.MODEL == "FCN":
    __C.RESOLUTION = 0.1    
    __C.SCALE=8
    __C.VOXEL_SHAPE=(800, 800, 40)
    __C.X=(0, 80)
    __C.Y=(-40, 40)
    __C.Z=(-2.5, 1.5)
elif __C.MODEL == "DENSENET":
    __C.RESOLUTION = 0.25
    __C.SCALE=4
    __C.VOXEL_SHAPE=(320, 320, 16)
    __C.X=(0, 80)
    __C.Y=(-40, 40)
    __C.Z=(-2.5, 1.5)
# GPU CONFIGURATION
__C.GPU_MEMORY_FRACTION=1.0
__C.GPU_AVAILABLE='0'                       # This line is with no use. If you want to set GPU Number, please set the environment variable by ``` export CUDA_VISIBLE_DEVICES='1' ```
__C.GPU_USE_COUNT = len(__C.GPU_AVAILABLE.split(','))
# UTILS
__C.CORNER2CENTER_AVG = True                # average version or max version
# EVAL & TRAINING
__C.LOAD_CHECKPT = "checkpoint-00111361"    # In train.py, it will be initilized from this checkpoint ; In test.py, it will test this checkpoint.
# TRAINING
__C.VALIDATE_INTERVAL = 10                  
# EVAL
__C.EVAL_THRESHOLD = 0.8                    
__C.BV_LOG_FACTOR = 4
# NMS
__C.ENABLE_NMS = True                       # Enable NMS or not.
__C.RPN_NMS_POST_TOPK = 20
__C.RPN_NMS_THRESH = 0.1
# for camera and lidar coordination convert
if __C.DATA_SETS_TYPE == 'kitti':
    # cal mean from train set
    __C.MATRIX_P2 = ([[719.787081,    0.,            608.463003, 44.9538775],
                      [0.,            719.787081,    174.545111, 0.1066855],
                      [0.,            0.,            1.,         3.0106472e-03],
                      [0.,            0.,            0.,         0]])

    # cal mean from train set
    __C.MATRIX_T_VELO_2_CAM = ([
        [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
        [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
        [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
        [0, 0, 0, 1]
    ])
    # cal mean from train set
    __C.MATRIX_R_RECT_0 = ([
        [0.99992475, 0.00975976, -0.00734152, 0],
        [-0.0097913, 0.99994262, -0.00430371, 0],
        [0.00729911, 0.0043753, 0.99996319, 0],
        [0, 0, 0, 1]
    ])

