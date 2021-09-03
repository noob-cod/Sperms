"""
@Date: 2021/9/3 下午2:36
@Author: Chen Zhang
@Brief: 训练
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint

from UNet.UNet import UNet
from UNetpp.UNetpp import UNetPP

from config import cfg


patterns = {
    'segmentation': {
        'UNet': UNet,
        'UNet++': UNetPP
    },
    'detection': {

    }
}

# =========================================== #
# 数据预处理
# =========================================== #


# =========================================== #
# 构建数据集
# =========================================== #
training_set = None
validation_set = None

# =========================================== #
# 构建神经网络
# =========================================== #
if cfg.TRAIN.PATTERN == 'segmentation':
    my_model = patterns[cfg.TRAIN.PATTERN][cfg.TRAIN.MODEL](
        input_shape=cfg.TRAIN.UNET_INPUT_SHAPE,
        filter_root=cfg.TRAIN.UNET_FILTER_ROOT,
        depth=cfg.TRAIN.UNET_DEPTH,
        out_dim=cfg.TRAIN.UNET_OUT_DIM,
        activation_type=cfg.TRAIN.UNET_ACTIVATION,
        kernel_initializer_type=cfg.TRAIN.UNET_KERNEL_INITIALIZER,
        batch_norm=cfg.TRAIN.UNET_BATCH_NORM
    )
    model = my_model.get_model()
elif cfg.TRAIN.PATTERN == 'detection':
    model = None
else:
    model = None
    assert 1 == 0, '不存在的模式!'

# =========================================== #
# 设置callbacks
# =========================================== #
callbacks = []

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

# =========================================== #
# 编译神经网络
# =========================================== #
model.compile(
    optimizer=cfg.TRAIN.OPTIMIZER,
    loss=cfg.TRAIN.LOSS,
    metrics=cfg.TRAIN.METRICS
)

# =========================================== #
# 训练
# =========================================== #
model.fit(
    training_set,
    epochs=cfg.TRAIN.EPOCH,
    callbacks=callbacks,
    validation_data=validation_set
)
