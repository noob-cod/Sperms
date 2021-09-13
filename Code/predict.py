"""
@Date: 2021/9/12 上午10:43
@Author: Chen Zhang
@Brief: 对测试图像进行推理，输出推理结果
"""
import os
import cv2
import tensorflow as tf

from UNet.UNet import UNet
from UNetpp.UNetpp import UNetPP
from utils.datamaker.TFRecorder import TFRecorder

from config import cfg


patterns = {
    'segmentation': {
        'UNet': UNet,
        'UNet++': UNetPP
    },

    'detection': {

    }
}

if cfg.TRAIN.PATTERN == 'segmentation':
    my_model = patterns[cfg.TRAIN.PATTERN][cfg.TRAIN.MODEL](
        input_shape=cfg.TEST.INPUT_SHAPE,
        filter_root=cfg.TRAIN.UNET_FILTER_ROOT,
        depth=cfg.TRAIN.UNET_DEPTH,
        out_dim=cfg.TRAIN.UNET_OUT_DIM,
        activation_type=cfg.TRAIN.UNET_ACTIVATION,
        kernel_initializer_type=cfg.TRAIN.UNET_KERNEL_INITIALIZER,
        batch_norm=cfg.TRAIN.UNET_BATCH_NORMALIZATION
    )
    model = my_model.get_model()
elif cfg.TRAIN.PATTERN == 'detection':
    model = None
else:
    model = None


checkpoint_dir = 'Filters-16_BatchSize-8_Depth-5_Epoch-20_Loss-dice_OPTIMIZER-Adam'
model_name = 'Model-epoch_11-acc_0.9671-val_acc_0.9675.h5'

model.load_weights('/home/bmp/ZC/Sperms/Checkpoint/UNet/' +
                   checkpoint_dir + '/'
                   'models/' +
                   model_name)

first_path = os.path.join(cfg.TEST.SAVE_PATH, checkpoint_dir)  # 基于超参数建立上层文件夹
second_path = os.path.join(first_path, model_name)  # 基于具体模型建立实际输出文件夹

img_names = os.listdir(cfg.DATA.TEST_IMG)

for name in img_names:
    img = tf.io.read_file(os.path.join(cfg.DATA.TEST_IMG, name))
    img = tf.io.decode_image(img) / 255
    img = tf.expand_dims(img, axis=0)
    print(img.shape)
    result = model.predict(img)
    print(result.shape)

    result = tf.squeeze(result, [0])
    result = tf.round(result * 255)
    result = tf.cast(result, dtype=tf.uint8)
    print(result.shape)

    pred_BG = result[:, :, 0]  # 顶体未发生反应
    pred_BG = tf.expand_dims(pred_BG, axis=-1)
    pred_BG = tf.io.encode_png(pred_BG)
    tf.io.write_file(os.path.join(second_path, name + '_BG.png'), pred_BG)

    pred_AI = result[:, :, 1]  # 顶体未发生反应
    pred_AI = tf.expand_dims(pred_AI, axis=-1)
    pred_AI = tf.io.encode_png(pred_AI)
    tf.io.write_file(os.path.join(second_path, name + '_AI.png'), pred_AI)

    pred_AR = result[:, :, 2]  # 顶体发生反应
    pred_AR = tf.expand_dims(pred_AR, axis=-1)
    pred_AR = tf.io.encode_png(pred_AR)
    tf.io.write_file(os.path.join(second_path, name + '_AR.png'), pred_AR)

    pred_EX = result[:, :, 3]  # 顶体异常
    pred_EX = tf.expand_dims(pred_EX, axis=-1)
    pred_EX = tf.io.encode_png(pred_EX)
    tf.io.write_file(os.path.join(second_path, name + '_EX.png'), pred_EX)

    pred_OTHERS = result[:, :, 4]  # 其他
    pred_OTHERS = tf.expand_dims(pred_OTHERS, axis=-1)
    pred_OTHERS = tf.io.encode_png(pred_OTHERS)
    tf.io.write_file(os.path.join(second_path, name[:-4] + '_OTHERS.png'), pred_OTHERS)
    break