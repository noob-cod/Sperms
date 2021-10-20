"""
@Date: 2021/9/5 上午10:16
@Author: Chen Zhang
@Brief:
"""
import tensorflow as tf

from tensorflow.keras.losses import Loss


class DiceLoss(Loss):

    def call(self, y_true, y_pred, smooth=1e-5, factor=(1, 1, 1, 1, 1)):
        """输入图像的格式为(H, W，C)"""
        # 计算交集和并集
        intersection = tf.reduce_sum(tf.math.multiply(y_true, y_pred), axis=[0, 1])
        union = tf.reduce_sum(y_true, axis=[0, 1]) + tf.reduce_sum(y_pred, axis=[0, 1])
        # 计算dice
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice
        # return dice_loss
        return (factor[0] * dice_loss[0] +
                factor[1] * dice_loss[1] +
                factor[2] * dice_loss[2] +
                factor[3] * dice_loss[3] +
                factor[4] * dice_loss[4]) / 5


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
    print(DiceLoss()(true, pred))
    print(tf.keras.losses.BinaryCrossentropy(from_logits=True)(true, pred).numpy())
