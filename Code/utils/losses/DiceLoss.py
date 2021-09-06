"""
@Date: 2021/9/5 上午10:16
@Author: Chen Zhang
@Brief:
"""
import tensorflow as tf

from tensorflow.keras.losses import Loss


class DiceLoss(Loss):

    def call(self, y_true, y_pred, smooth=1e-5):
        """输入图像的格式为(H, W，C)"""
        # 计算交集和并集
        intersection = tf.reduce_sum(tf.math.multiply(y_true, y_pred))
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        # 计算dice
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return tf.reduce_mean(dice)


if __name__ == '__main__':
    true = [[[1.0, 1.0],
             [0.0, 0.0]],
            [[0.0, 0.0],
             [1.0, 1.0]]]
    pred = [[[0.8, 0.2],
             [0.5, 0.5]],
            [[0.5, 0.5],
             [0.5, 0.5]]]
    print(DiceLoss()(true, pred))