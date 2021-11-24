"""
@Date: 2021/9/5 上午10:16
@Author: Chen Zhang
@Brief:
"""
import tensorflow as tf

from tensorflow.keras.losses import Loss


class DiceLoss(Loss):

    def call(self, y_true, y_pred):
        """输入图像的格式为(H, W，C)"""
        y_pred = tf.convert_to_tensor(y_pred)
        # print('inner_pred num:', y_pred.shape)
        smooth = 1e-5
        # print()
        # print('multiply:')
        # print(tf.math.multiply(y_true, y_pred))
        # print()
        # 计算交集和并集
        intersection = tf.reduce_sum(tf.math.multiply(y_true, y_pred), axis=[-3, -2])
        # print('intersection:')
        # print(intersection)
        # print()
        union = tf.reduce_sum(y_true, axis=[-3, -2]) + tf.reduce_sum(y_pred, axis=[-3, -2])
        # print('union:')
        # print(union)
        # print()
        # 计算dice
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice
        # print('dice loss:')
        # print(dice_loss)
        # print()

        # return dice_loss
        if self.reduction == tf.keras.losses.Reduction.NONE:
            return dice_loss
        elif self.reduction == tf.keras.losses.Reduction.SUM:
            return tf.reduce_sum(dice_loss)
        else:
            return tf.reduce_mean(dice_loss)


if __name__ == '__main__':
    true = tf.constant(
        [[[1.0, 1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0]],
         [[1.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0]]]
    )
    print('true shape:', true.shape)
    pred = tf.constant(
        [[[0.8, 0.7, 0.3, 0.3, 0.2], [0.6, 0.7, 0.2, 0.1, 0.7]],
         [[1.0, 0.4, 0.5, 0.5, 0.1], [0.4, 0.6, 0.7, 0.6, 0.2]]]
    )
    print('pred shape:', pred.shape)
    # print('dice loss:', DiceLoss()(true, pred))
    # print('dice loss SUM:', DiceLoss(reduction=tf.keras.losses.Reduction.SUM)(true, pred).numpy())
    # print('dice loss NONE:', DiceLoss(reduction=tf.keras.losses.Reduction.NONE)(true, pred).numpy())
    print(DiceLoss()(true, [pred, pred, pred]))

    print('bce None')
    print(tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(true, [pred, pred, pred]))
    print('bce SUM')
    print(tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)(true, [pred, pred, pred]))
    print('bce')
    print(tf.keras.losses.BinaryCrossentropy()(true, [pred, pred, pred]))
    # print(tf.keras.losses.BinaryCrossentropy()(true, pred))
