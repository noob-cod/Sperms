"""
@Date: 2021/10/20 下午8:32
@Author: Chen Zhang
@Brief:
"""
import tensorflow as tf

from tensorflow.keras.losses import Loss
from Code.config import cfg


class DeepSupLoss(Loss):

    def call(self, y_true, y_pred, dice_smooth=1e-5):
        # 计算binary cross-entropy
        if cfg.TRAIN.DISTRIBUTE_FLAG:
            bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
        else:
            bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

        # 计算dice
        # 交集和并集
        if not isinstance(y_pred, list):
            intersection = tf.reduce_sum(tf.math.multiply(y_true, y_pred), axis=[0, 1])
            union = tf.reduce_sum(y_true, axis=[0, 1]) + tf.reduce_sum(y_pred, axis=[0, 1])

            dice = (2.0 * intersection + dice_smooth) / (union + dice_smooth)
            dice_loss = tf.reduce_sum(1-dice) / 5  # sum(1 - dice) / 5
        else:
            nums = len(y_pred)
            # print('nums:', nums)
            dice_loss = tf.constant(0, dtype=tf.float32)
            for i in range(nums):
                intersection = tf.reduce_sum(tf.math.multiply(y_true, y_pred[i]), axis=[0, 1])
                union = tf.reduce_sum(y_true, axis=[0, 1]) + tf.reduce_sum(y_pred[i], axis=[0, 1])

                dice = (2.0 * intersection + dice_smooth) / (union + dice_smooth)
                # print('dice:', dice)
                dice_loss += tf.reduce_sum(1-dice) / 5  # sum(1-dice) / 5
                # print('dice loss:', 1-dice)
                # print('total dice loss:', dice_loss)
                # print()
            dice_loss = dice_loss / nums

        total_loss = 0.5 * (bce + dice_loss)

        return total_loss


if __name__ == '__main__':
    true = tf.constant(
        [[[1.0, 1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0]],
         [[1.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0]]]
    )
    print(true.shape)
    pred = tf.constant(
        [[[0.8, 0.7, 0.3, 0.3, 0.2], [0.6, 0.7, 0.2, 0.1, 0.7]],
         [[1.0, 0.4, 0.5, 0.5, 0.1], [0.4, 0.6, 0.7, 0.6, 0.2]]]
    )
    print(pred.shape)
    print(DeepSupLoss()(true, [pred, pred+0.1, pred-0.1]))
