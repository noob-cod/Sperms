"""
@Date: 2021/10/20 下午7:54
@Author: Chen Zhang
@Brief:
"""
import datetime
import tensorflow as tf
from easydict import EasyDict


__C = EasyDict()
cfg = __C

# 数据集相关参数
__C.DATA = EasyDict()

# 训练相关参数
__C.TRAIN = EasyDict()

# 模型相关参数
__C.MODEL = EasyDict()

# 训练回调相关参数
__C.CALLBACKS = EasyDict()

# 测试相关参数
__C.TEST = EasyDict()

# =============================================== #
# 训练参数设置
# =============================================== #
# 损失函数
__C.TRAIN.LOSS = 'bcedice'

# 优化器
__C.TRAIN.OPTIMIZER = 'Adam'

# 初始学习速率
__C.TRAIN.LR = 0.01

# 监控窗口
__C.TRAIN.METRICS = [
    'miou'
]
# 轮次
__C.TRAIN.EPOCH = 100

# 实际训练样本占总训练样本的比例
__C.TRAIN.TRAINING_RATIO = 0.8

# 训练集的batch_size，多卡训练时为每张卡的batch size
__C.TRAIN.BATCH_SIZE = 8

# 是否进行分布式训练
__C.TRAIN.DISTRIBUTE_FLAG = True

# 指定gpu
__C.TRAIN.DISTRIBUTE_DEVICES = ['/gpu:0', '/gpu:1']

# =============================================== #
# 模型参数设置
# =============================================== #
# 模型的类别
__C.MODEL.PATTERN = 'segmentation'

# 模型选择
__C.MODEL.TYPE = 'UNet++'

# 输入尺寸
__C.MODEL.UNET_INPUT_SHAPE = (256, 256, 3)

# 第1个Floor的卷积核数量
__C.MODEL.UNET_FILTER_ROOT = 16

# 深度（Floor数量）
__C.MODEL.UNET_DEPTH = 5

# 输出的通道数（包含背景后类的数量）
__C.MODEL.UNET_OUT_DIM = 5

# 激活函数
__C.MODEL.UNET_ACTIVATION = 'relu'

# 卷积核初始化方式
__C.MODEL.UNET_KERNEL_INITIALIZER = 'he_normal'

# 是否带有dropout层，值为0时表示不包含dropout
__C.MODEL.UNET_DROPOUT = 0

# 是否带有批正则化层，默认包含
__C.MODEL.UNET_BATCH_NORMALIZATION = True

# 是否启用深度监督
__C.MODEL.UNETPP_DEEP_SUPERVISION = True

# =============================================== #
# Dataset设置
# =============================================== #

# 训练集图片路径
__C.DATA.TRAIN_IMG = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/training_set/img'

# 训练集mask路径
__C.DATA.TRAIN_MSK = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/training_set/mask'

# 测试集图片路径
__C.DATA.TEST_IMG = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/test_set/src/img'

# 测试集mask路径
__C.DATA.TEST_MSK = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/test_set/src/mask'

# 训练集tfrecord路径
__C.DATA.TRAINING_TFRECORD = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/training_set/tfrecord'

# 训练集打乱时的缓存区大小
__C.DATA.SHUFFLE_BUFFER_SIZE = 500

# =============================================== #
# 设置callbacks
# =============================================== #

# tf.keras.callbacks.LRSchedule设置
__C.CALLBACKS.LR_SCHEDULER = False
__C.CALLBACKS.LR_SCHEDULE_ID = 0
__C.CALLBACKS.LR_SCHEDULER_VERBOSE = 1

# tf.keras.callbacks.EarlyStopping设置
__C.CALLBACKS.EARLY_STOPPING = False
__C.CALLBACKS.EARLY_STOPPING_PATIENCE = 10
__C.CALLBACKS.EARLY_STOPPING_VERBOSE = 1

# tf.keras.callbacks.ReduceLRONPlateau设置
__C.CALLBACKS.REDUCE_LR = False
__C.CALLBACKS.REDUCE_LR_FACTOR = 0.1
__C.CALLBACKS.REDUCE_LR_PATIENCE = 10
__C.CALLBACKS.REDUCE_LR_MIN_LR = 0.00001
__C.CALLBACKS.REDUCE_LR_VERBOSE = 1

# tf.keras.callbacks.ModelCheckpoint设置
__C.CALLBACKS.CHECK_POINT_DIR = '/home/bmp/ZC/Sperms/Checkpoint/' + __C.MODEL.TYPE + '/' + \
                                'Filters-' + str(__C.MODEL.UNET_FILTER_ROOT) + \
                                '_BatchSize-' + str(__C.TRAIN.BATCH_SIZE) + \
                                '_Depth-' + str(__C.MODEL.UNET_DEPTH) + \
                                '_Epoch-' + str(__C.TRAIN.EPOCH) + \
                                '_Loss-' + __C.TRAIN.LOSS + \
                                '_Optimizer-' + __C.TRAIN.OPTIMIZER + \
                                '_InitLR-' + str(__C.TRAIN.LR) + \
                                '_Time-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")  # 年-月-日-时-分
__C.CALLBACKS.CHECK_POINT = True
__C.CALLBACKS.CHECK_POINT_FILEPATH = __C.CALLBACKS.CHECK_POINT_DIR + '/models' + \
                                     '/Model-epoch_{epoch:02d}-acc_{accuracy:.4f}-val_acc_{val_accuracy:.4f}-' \
                                     '.h5'
__C.CALLBACKS.CHECK_POINT_MONITOR = 'val_loss'
__C.CALLBACKS.CHECK_POINT_VERBOSE = 1
__C.CALLBACKS.CHECK_POINT_SAVE_BEST = False
__C.CALLBACKS.CHECK_POINT_SAVE_WEIGHTS = True
__C.CALLBACKS.CHECK_POINT_MODE = 'auto'
__C.CALLBACKS.CHECK_POINT_SAVE_FREQ = 'epoch'

# tf.keras.callbacks.TensorBoard设置
__C.CALLBACKS.TENSOR_BOARD = True
__C.CALLBACKS.TENSOR_BOARD_LOG_DIR = __C.CALLBACKS.CHECK_POINT_DIR + '/logs'
__C.CALLBACKS.TENSOR_BOARD_HISTOGRAM_FREQ = 0
__C.CALLBACKS.TENSOR_BOARD_GRAPH = False
__C.CALLBACKS.TENSOR_BOARD_UPDATE_FREQ = 'epoch'

# =============================================== #
# Test设置
# =============================================== #

# 测试输入
__C.TEST.INPUT_SHAPE = (896, 1024, 3)  # (2448, 3264, 3)

# 测试输出保存路径
__C.TEST.SAVE_PATH = '/home/bmp/ZC/Sperms/Outputs/' + __C.MODEL.TYPE
