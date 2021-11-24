"""
@Date: 2021/10/20 下午8:32
@Author: Chen Zhang
@Brief:
"""
import tensorflow as tf

from tensorflow.keras.losses import Loss
from Code.utils.losses.DiceLoss import DiceLoss
from Code.config import cfg


class DeepSupLoss(Loss):

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        # print(y_pred.shape)

        b, h, w, c = y_pred.shape

        # 计算bce和dice
        if cfg.TRAIN.DISTRIBUTE_FLAG:
            bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)(y_true, y_pred)
            # print(y_pred.shape)
            # if cfg.MODEL.TYPE == 'UNet++' and cfg.MODEL.UNETPP.DEEP_SUPERVISION:
            #     bce_factor = tf.convert_to_tensor(b*h*w, dtype=tf.float32)
            # else:
            bce_factor = tf.convert_to_tensor(h*w, dtype=tf.float32)
            # # print('factor_0:', bce_factor)
            bce = tf.math.divide(bce, bce_factor)
            dice_loss = DiceLoss(reduction=tf.keras.losses.Reduction.SUM)(y_true, y_pred)
            # if cfg.MODEL.TYPE == 'UNet++' and cfg.MODEL.UNETPP.DEEP_SUPERVISION:
            #     dice_factor = tf.convert_to_tensor(b * c, dtype=tf.float32)
            # else:
            dice_factor = tf.convert_to_tensor(c, dtype=tf.float32)
            dice_loss = tf.math.divide(dice_loss, dice_factor)
        else:
            bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
            dice_loss = DiceLoss()(y_true, y_pred)

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
    # bce = tf.keras.losses.BinaryCrossentropy()(true, pred)
    # print('bce:', tf.math.reduce_mean(bce))
    # dice = DiceLoss()(true, pred)
    # print('dice:', dice)
    # print('bce_dice:', (bce + dice) / 2)
    print(DeepSupLoss()(true, [pred, pred, pred]))

    # print('BCE:')
    # print(tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(true, [pred, pred, pred]))
    # print()
    # print('Dice:')
    # print(DiceLoss(reduction=tf.keras.losses.Reduction.NONE)(true, [pred, pred, pred]))
