"""
@Date: 2021/9/5 上午10:16
@Author: Chen Zhang
@Brief:
"""
import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.keras.losses import Loss


class DiceLoss(Loss):

    def call(self, y_true, y_pred, axis=(1, 2), smooth=1e-5):
        # 计算交集和并集
        intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
        union = tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_pred, axis=axis)
        # 计算dice
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return tf.reduce_mean(dice)


if __name__ == '__main__':
    pass
