"""
@Date: 2021/9/3 下午2:37
@Author: Chen Zhang
@Brief:
"""
from easydict import EasyDict


__C = EasyDict()
cfg = __C

__C.DATA = EasyDict()
__C.TRAIN = EasyDict()
__C.CALLBACKS = EasyDict()

# =============================================== #
# 模型编译与训练参数设置
# =============================================== #

# 模型的类别
__C.TRAIN.PATTERN = 'segmentation'

# 损失函数
__C.TRAIN.LOSS = 'binary_crossentropy'

# 优化器
__C.TRAIN.OPTIMIZER = 'Adam'

# 监控窗口
__C.TRAIN.METRICS = ['accuracy']

# 轮次
__C.TRAIN.EPOCH = 10

# =============================================== #
# UNet & UNet++ Training Options
# =============================================== #

# 模型选择
__C.TRAIN.MODEL = 'UNet'

# 输入尺寸
__C.TRAIN.UNET_INPUT_SHAPE = (256, 256, 3)

# 第1个Floor的卷积核数量
__C.TRAIN.UNET_FILTER_ROOT = 16

# 深度（Floor数量）
__C.TRAIN.UNET_DEPTH = 5

# 输出的通道数（包含背景后类的数量）
__C.TRAIN.UNET_OUT_DIM = 5

# 激活函数
__C.TRAIN.UNET_ACTIVATION = 'relu'

# 卷积核初始化方式
__C.TRAIN.UNET_KERNEL_INITIALIZER = 'he_normal'

# 是否带有dropout层，值为0时表示不包含dropout
__C.TRAIN.UNET_DROPOUT = 0

# 是否带有批正则化层，默认包含
__C.TRAIN.UNET_BATCH_NORMALIZATION = True


# =============================================== #
# 设置callbacks
# =============================================== #

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
__C.CALLBACKS.CHECK_POINT = True
__C.CALLBACKS.CHECK_POINT_FILEPATH = 'Model-EPOCH:{epoch:02d}-ACC:{acc:.4f}-VAL_ACC:{val_acc:.4f}.h5'
__C.CALLBACKS.CHECK_POINT_MONITOR = 'val_loss'
__C.CALLBACKS.CHECK_POINT_VERBOSE = 1
__C.CALLBACKS.CHECK_POINT_SAVE_BEST = False
__C.CALLBACKS.CHECK_POINT_SAVE_WEIGHTS = True
__C.CALLBACKS.CHECK_POINT_MODE = 'auto'
__C.CALLBACKS.CHECK_POINT_SAVE_FREQ = 'epoch'

# tf.keras.callbacks.TensorBoard设置
__C.CALLBACKS.TENSOR_BOARD = True
__C.CALLBACKS.TENSOR_BOARD_LOG_DIR = '/home/bmp/ZC/Sperms/Checkpoint/UNet/logs'
__C.CALLBACKS.TENSOR_BOARD_HISTOGRAM_FREQ = 0
__C.CALLBACKS.TENSOR_BOARD_GRAPH = False
__C.CALLBACKS.TENSOR_BOARD_UPDATE_FREQ = 'epoch'

# =============================================== #
# Dataset设置
# =============================================== #

# 训练集图片路径
__C.DATA.TRAIN_IMG = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/training_set/img'

# 训练集mask路径
__C.DATA.TRAIN_MSK = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/training_set/mask'

# 测试集图片路径
__C.DATA.TEST_IMG = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/test/src/img'

# 测试集mask路径
__C.DATA.TEST_MSK = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/test/src/mask'

# 训练集tfrecord路径
__C.DATA.TRAINING_TFRECORD = '/home/bmp/ZC/Sperms/dataset/UNet_dataset/training_set/tfrecord'

# 训练集打乱时的缓存区大小
__C.DATA.SHUFFLE_BUFFER_SIZE = 1000

# 实际训练样本占总训练样本的比例
__C.DATA.TRAINING_RATIO = 0.8

# 训练集的batch_size
__C.DATA.BATCH_SIZE = 4
