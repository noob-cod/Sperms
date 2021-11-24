"""
@Date: 2021/11/24 下午7:00
@Author: Chen Zhang
@Brief:
"""
import tensorflow as tf

from tensorflow.python.keras import backend
from tensorflow.python.keras.metrics import MeanMetricWrapper
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import math_ops


def dice(y_true, y_pred, channel_id):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=y_true.dtype)

    # 提取指定通道
    if y_true.shape.ndims == 4:
        y_true = y_true[:, :, :, channel_id]  # 'channel last'
    elif y_true.shape.ndims == 3:
        y_true = y_true[:, :, channel_id]
    else:
        pass

    if y_pred.shape.ndims == 4:
        y_pred = y_pred[:, :, :, channel_id]
    elif y_pred.shape.ndims == 3:
        y_pred = y_pred[:, :, channel_id]
    else:
        pass

    smooth = 1e-5
    [y_pred, y_true], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values([y_pred, y_true])
    y_pred.shape.assert_is_compatible_with(y_true.shape)
    if y_true.dtype != y_pred.dtype:
        y_pred = math_ops.cast(y_pred, y_true.dtype)

    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0, 1])
    union = tf.reduce_sum(tf.multiply(y_true, y_true)) + tf.reduce_sum(tf.multiply(y_pred, y_pred))

    return math_ops.cast(math_ops.div_no_nan(2.0 * intersection + smooth, union + smooth), backend.floatx())


class Dice(MeanMetricWrapper):

    def __init__(self, name, dtype=None, channel_id=0):
        super(Dice, self).__init__(dice, name, dtype=dtype, channel_id=channel_id)


class AiDice(Dice):

    def __init__(self, name='ai_dice', dtype=None, channel_id=1):
        super(AiDice, self).__init__(name=name, dtype=dtype, channel_id=channel_id)


class ArDice(Dice):

    def __init__(self, name='ar_dice', dtype=None, channel_id=2):
        super(ArDice, self).__init__(name=name, dtype=dtype, channel_id=channel_id)


class BgDice(Dice):

    def __init__(self, name='bg_dice', dtype=None, channel_id=0):
        super(BgDice, self).__init__(name=name, dtype=dtype, channel_id=channel_id)


class NeglectDice(Dice):

    def __init__(self, name='neglect_dice', dtype=None, channel_id=3):
        super(NeglectDice, self).__init__(name=name, dtype=dtype, channel_id=channel_id)


class OthersDice(Dice):

    def __init__(self, name='others_dice', dtype=None, channel_id=4):
        super(OthersDice, self).__init__(name=name, dtype=dtype, channel_id=channel_id)


if __name__ == '__main__':
    gt = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 0, 0],
          [0, 0, 0, 1, 1, 0, 0, 0, 0],
          [0, 0, 0, 1, 1, 0, 0, 0, 0],
          [0, 0, 0, 1, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0]]

    pd = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 0, 0],
          [0, 0, 0, 1, 1, 0, 0, 0, 0],
          [0, 0, 0, 1, 1, 0, 0, 0, 0],
          [0, 0, 0, 1, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0]]

    m = Dice('Dice', 0)
    m.update_state(gt, pd)
    print(m.result().numpy())
