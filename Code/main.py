"""
@Date: 2021/10/20 下午7:51
@Author: Chen Zhang
@Brief: 支持单机多卡分布式训练
"""
import os
import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from UNet.UNet import UNet
from UNetpp.UNetPP import UNetPP
from utils.datamaker.TFRecorder import TFRecorder
from utils.losses.DiceLoss import DiceLoss
from utils.losses.FocalLoss import FocalLoss
from utils.losses.UNetPPLoss import DeepSupLoss
from utils.lr_schedule import LRScheduleManager

from config import cfg

strategy = None
GLOBAL_BATCH_SIZE = None
if cfg.TRAIN.DISTRIBUTE_FLAG:
    # 创建新的分布式训练策略
    strategy = tf.distribute.MirroredStrategy(devices=cfg.TRAIN.DISTRIBUTE_DEVICES)

    # 全局批尺寸 = 单卡批尺寸 * 卡数量
    GLOBAL_BATCH_SIZE = cfg.TRAIN.BATCH_SIZE * strategy.num_replicas_in_sync

PATTERNS = {
    'segmentation': {
        'UNet': UNet,
        'UNet++': UNetPP
    },
    'detection': {

    }
}

# =========================================== #
# 构建数据集
# =========================================== #
# 记录所有训练用的tfrecord文件路径
tf_record_files = []
for name in os.listdir(cfg.DATA.TRAINING_TFRECORD):
    tf_record_files.append(os.path.join(cfg.DATA.TRAINING_TFRECORD, name))

# 计算参与训练的数据的数量
train_set_nums = None
if cfg.TRAIN.TRAINING_RATIO:
    train_set_nums = round(len(os.listdir(cfg.DATA.TRAIN_IMG)) * cfg.TRAIN.TRAINING_RATIO)

# 若不划分验证集，则validation_set为None
training_set, validation_set = TFRecorder().read_from_tfrecord(tf_record_files, train_set_nums=train_set_nums)

# 生成batch
if cfg.TRAIN.DISTRIBUTE_FLAG:
    training_set = training_set.batch(batch_size=GLOBAL_BATCH_SIZE)
    if validation_set:
        validation_set = validation_set.batch(batch_size=GLOBAL_BATCH_SIZE)
else:
    training_set = training_set.batch(batch_size=cfg.TRAIN.BATCH_SIZE)
    if validation_set:
        validation_set = validation_set.batch(batch_size=cfg.TRAIN.BATCH_SIZE)

# =========================================== #
# 构建神经网络
# =========================================== #
if cfg.MODEL.PATTERN == 'segmentation':
    if PATTERNS[cfg.MODEL.PATTERN] == 'UNet++':
        my_model = PATTERNS[cfg.MODEL.PATTERN][cfg.MODEL.TYPE](
            input_shape=cfg.MODEL.UNET_INPUT_SHAPE,
            filter_root=cfg.MODEL.UNET_FILTER_ROOT,
            depth=cfg.MODEL.UNET_DEPTH,
            out_dim=cfg.MODEL.UNET_OUT_DIM,
            activation_type=cfg.MODEL.UNET_ACTIVATION,
            kernel_initializer_type=cfg.MODEL.UNET_KERNEL_INITIALIZER,
            batch_norm=cfg.MODEL.UNET_BATCH_NORMALIZATION,
            deep_supervision=cfg.MODEL.UNETPP_DEEP_SUPERVISION
        )
    else:
        my_model = PATTERNS[cfg.MODEL.PATTERN][cfg.MODEL.TYPE](
            input_shape=cfg.MODEL.UNET_INPUT_SHAPE,
            filter_root=cfg.MODEL.UNET_FILTER_ROOT,
            depth=cfg.MODEL.UNET_DEPTH,
            out_dim=cfg.MODEL.UNET_OUT_DIM,
            activation_type=cfg.MODEL.UNET_ACTIVATION,
            kernel_initializer_type=cfg.MODEL.UNET_KERNEL_INITIALIZER,
            batch_norm=cfg.MODEL.UNET_BATCH_NORMALIZATION
        )
elif cfg.MODEL.PATTERN == 'detection':
    my_model = None
    assert 1 == 0, '不存在的模式!'
else:
    my_model = None
    assert 1 == 0, '不存在的模式!'

if cfg.TRAIN.DISTRIBUTE_FLAG:
    with strategy.scope():
        model = my_model.get_model()
else:
    model = my_model.get_model()

# =========================================== #
# 编译神经网络
# =========================================== #
loss_dict = {
    'bce': 'binary_crossentropy',
    'dice': DiceLoss,
    'fcl': FocalLoss,
    'bcedice': DeepSupLoss
}

optimizer_dict = {
    'adam': tf.keras.optimizers.Adam
}

metric_dict = {
    'precision': tf.keras.metrics.Precision,
    'miou': tf.keras.metrics.MeanIoU
}

if cfg.TRAIN.DISTRIBUTE_FLAG:
    with strategy.scope():
        # loss
        loss = loss_dict[cfg.TRAIN.LOSS.lower()]()

        # optimizer
        optimizer = optimizer_dict[cfg.TRAIN.OPTIMIZER.lower()](learning_rate=cfg.TRAIN.LR)

        # metrics
        metrics = ['accuracy']
        for item in cfg.TRAIN.METRICS:
            if item != 'miou':
                metrics.append(metric_dict[item]())
            else:
                metrics.append(metric_dict[item](num_classes=5))

        # compile
        model.compile(
            optimizer=cfg.TRAIN.OPTIMIZER,
            loss=loss,
            metrics=metrics
        )
else:
    # loss
    loss = loss_dict[cfg.TRAIN.LOSS.lower()]()

    # metrics
    metrics = ['accuracy']
    for item in cfg.TRAIN.METRICS:
        if item != 'miou':
            metrics.append(metric_dict[item]())
        else:
            metrics.append(metric_dict[item](num_classes=5))

    # compile
    model.compile(
        optimizer=cfg.TRAIN.OPTIMIZER,
        loss=loss,
        metrics=metrics
    )

# =========================================== #
# 设置callbacks
# =========================================== #
callbacks = []

if cfg.CALLBACKS.LR_SCHEDULER:
    callbacks.append(LearningRateScheduler(
        schedule=LRScheduleManager().get_schedule(id=cfg.CALLBACKS.LR_SCHEDULE_ID),
        verbose=cfg.CALLBACKS.LR_SCHEDULER_VERBOSE
    ))

if cfg.CALLBACKS.EARLY_STOPPING:
    callbacks.append(EarlyStopping(
        patience=cfg.CALLBACKS.EARLY_STOPPING_PATIENCE,
        verbose=cfg.CALLBACKS.EARLY_STOPPING_VERBOSE
    ))

if cfg.CALLBACKS.REDUCE_LR:
    callbacks.append(ReduceLROnPlateau(
        factor=cfg.CALLBACKS.REDUCE_LR_FACTOR,
        patience=cfg.CALLBACKS.REDUCE_LR_PATIENCE,
        min_lr=cfg.CALLBACKS.REDUCE_LR_MIN,
        verbose=cfg.CALLBACKS.REDUCE_LR_VERBOSE
    ))

if cfg.CALLBACKS.CHECK_POINT:
    callbacks.append(ModelCheckpoint(
        filepath=cfg.CALLBACKS.CHECK_POINT_FILEPATH,
        monitor=cfg.CALLBACKS.CHECK_POINT_MONITOR,
        verbose=cfg.CALLBACKS.CHECK_POINT_VERBOSE,
        save_best_only=cfg.CALLBACKS.CHECK_POINT_SAVE_BEST,
        save_weights_only=cfg.CALLBACKS.CHECK_POINT_SAVE_WEIGHTS,
        mode=cfg.CALLBACKS.CHECK_POINT_MODE,
        save_freq=cfg.CALLBACKS.CHECK_POINT_SAVE_FREQ
    ))

if cfg.CALLBACKS.TENSOR_BOARD:
    callbacks.append(TensorBoard(
        log_dir=cfg.CALLBACKS.TENSOR_BOARD_LOG_DIR,
        histogram_freq=cfg.CALLBACKS.TENSOR_BOARD_HISTOGRAM_FREQ,
        write_graph=cfg.CALLBACKS.TENSOR_BOARD_GRAPH,
        update_freq=cfg.CALLBACKS.TENSOR_BOARD_UPDATE_FREQ
    ))

if not callbacks:
    callbacks = None
else:
    os.mkdir(path=cfg.CALLBACKS.CHECK_POINT_DIR)
    os.mkdir(path=cfg.CALLBACKS.CHECK_POINT_DIR + '/models')
    os.mkdir(path=cfg.CALLBACKS.CHECK_POINT_DIR + '/logs')

# =========================================== #
# 训练
# =========================================== #

model.fit(
    training_set,
    epochs=cfg.TRAIN.EPOCH,
    callbacks=callbacks,
    validation_data=validation_set
)
