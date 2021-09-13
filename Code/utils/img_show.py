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
model = cfg.TRAIN.MODEL

img_path = os.path.join(output_path, model)

img_files = os.listdir(img_path)

for img_file in img_files:
    img = tf.io.read_file(os.path.join(img_path, img_file))
    img = tf.io.decode_png(img)
    plt.imshow(img)
    plt.title(img_file)
    plt.show()
