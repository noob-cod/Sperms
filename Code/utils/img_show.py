"""
@Date: 2021/9/13 上午10:00
@Author: Chen Zhang
@Brief: 用于显示Outputs中的预测图像
"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from Code.config import cfg


output_path = '/home/bmp/ZC/Sperms/Outputs/'
model = cfg.MODEL.TYPE

img_path = os.path.join(output_path, model, 'Filters-16_BatchSize-16_Depth-5_Epoch-20_Loss-dice_OPTIMIZER-Adam/Model-epoch_20-acc_0.8598-val_acc_0.9204.h5')

img_files = os.listdir(img_path)

for img_file in img_files:
    img = tf.io.read_file(os.path.join(img_path, img_file))
    img = tf.io.decode_png(img)
    plt.imshow(img)
    plt.title(img_file)
    plt.show()
