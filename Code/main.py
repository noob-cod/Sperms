"""
@Date: 2021/10/20 下午7:51
@Author: Chen Zhang
@Brief: 支持单机多卡分布式训练
"""
import os
import csv
import codecs
import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.losses import BinaryCrossentropy
from UNet.UNet import UNet
from UNetpp.UNetPP import UNetPP
from TransUNet.TransUNet import TransUnet
from utils.datamaker.TFRecorder import TFRecorder
from utils.losses.DiceLoss import DiceLoss
from utils.losses.FocalLoss import FocalLoss
from utils.losses.UNetPPLoss import DeepSupLoss
from utils.lr_schedule import LRScheduleManager
from utils.metrics.Dice import AiDice, ArDice, BgDice, NeglectDice, OthersDice
from utils.metrics.IOU import AiIOU, ArIOU, BgIOU, NeglectIOU, OthersIOU

from config import cfg

strategy = None
GLOBAL_BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
BATCH_SIZE_PER_REPLICA = None
if cfg.TRAIN.DISTRIBUTE_FLAG:
    # 创建新的分布式训练策略
    strategy = tf.distribute.MirroredStrategy(devices=cfg.TRAIN.DISTRIBUTE_DEVICES)

    # 全局批尺寸 = 单卡批尺寸 * 卡数量
    assert GLOBAL_BATCH_SIZE % strategy.num_replicas_in_sync == 0, 'batch size与数量不匹配！'
    BATCH_SIZE_PER_REPLICA = GLOBAL_BATCH_SIZE // strategy.num_replicas_in_sync

PATTERNS = {
    'segmentation': {
        'UNet': UNet,
        'UNet++': UNetPP,
        'TransUNet': TransUnet
    }
}

# =========================================== #
# 构建数据集
# =========================================== #
print('Making dataset ...')
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
print('Done!')
print()

# =========================================== #
# 构建神经网络
# =========================================== #
print('Pattern is: ', cfg.MODEL.PATTERN)
print('Model type is: ', cfg.MODEL.TYPE)
my_model = PATTERNS[cfg.MODEL.PATTERN][cfg.MODEL.TYPE]()

print('Constructing model ...')
if cfg.TRAIN.DISTRIBUTE_FLAG:
    with strategy.scope():
        model = my_model.get_model()
else:
    model = my_model.get_model()
print('Done!')
print()

# =========================================== #
# 编译神经网络
# =========================================== #
print('Compiling model ...')
loss_dict = {
    'bce': BinaryCrossentropy,
    'dice': DiceLoss,
    'fcl': FocalLoss,
    'bcedice': DeepSupLoss
}

optimizer_dict = {
    'adam': tf.keras.optimizers.Adam,
    'sgd': tf.keras.optimizers.SGD
}

metric_dict = {
    'precision': tf.keras.metrics.Precision,
    'miou': tf.keras.metrics.MeanIoU,
    'ai_iou': AiIOU,
    'ar_iou': ArIOU,
    'bg_iou': BgIOU,
    'neglect_iou': NeglectIOU,
    'others_iou': OthersIOU,
    'ai_dice': AiDice,
    'ar_dice': ArDice,
    'bg_dice': BgDice,
    'neglect_dice': NeglectDice,
    'others_dice': OthersDice,
}

if cfg.TRAIN.DISTRIBUTE_FLAG:
    with strategy.scope():
        # loss
        loss = loss_dict[cfg.TRAIN.LOSS.lower()]()

        # optimizer
        optimizer = optimizer_dict[cfg.TRAIN.OPTIMIZER.lower()](cfg.TRAIN.OPTIMIZER_PARAMETERS)

        # metrics
        # metrics = ['accuracy']
        # for item in cfg.TRAIN.METRICS:
        #     if item != 'miou':
        #         metrics.append(metric_dict[item]())
        #     else:
        #         metrics.append(metric_dict[item](num_classes=5))
        metrics = ['accuracy']
        for item in cfg.TRAIN.METRICS:
            if item == 'miou':
                metrics.append(metric_dict[item](num_classes=2))
            else:
                metrics.append(metric_dict[item]())

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
            metrics.append(metric_dict[item](num_classes=2))

    # compile
    model.compile(
        optimizer=cfg.TRAIN.OPTIMIZER,
        loss=loss,
        metrics=metrics
    )
print('Done!')
print()

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
# 保存模型配置
# =========================================== #
print('正在保存模型配置...')
# 训练参数
training_info = [
    ['模型', 'Loss', 'Optimizer', 'Learning Rate', 'Epoch', 'Batch_size'],
    [cfg.MODEL.TYPE, cfg.TRAIN.LOSS, cfg.TRAIN.OPTIMIZER, cfg.TRAIN.LR, cfg.TRAIN.EPOCH, cfg.TRAIN.BATCH_SIZE]
]
# 模型参数
model_info = my_model.get_info()

# 合并训练参数和模型参数
training_info[0].extend(model_info[0])
training_info[1].extend(model_info[1])

# 写入配置
f = codecs.open(os.path.join(cfg.CALLBACKS.CHECK_POINT_DIR,
                             'ModelConfig.csv'), 'w', 'gbk')  # utf-8需要转gbk
writer = csv.writer(f)
for i in training_info:
    writer.writerow(i)
f.close()
print('Done!')
print()

# =========================================== #
# 训练
# =========================================== #

model.fit(
    training_set,
    epochs=cfg.TRAIN.EPOCH,
    callbacks=callbacks,
    validation_data=validation_set
)
