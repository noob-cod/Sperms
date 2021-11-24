"""
@Date: 2021/11/23 上午9:51
@Author: Chen Zhang
@Brief:  单类IOU的实现

IOU = TP / (TP + FP + FN)
"""
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops


class BinaryIOU(tf.keras.metrics.Metric):

    def __init__(self, name, channel_id, thresh=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.channel_id = channel_id  # 要计算iou的通道，每个通道包含一个类的预测结果
        self.thresh = thresh  # 二值化的阈值

        self.class_cm = self.add_weight(
            name='class_confusion_matrix',
            shape=(2, 2),
            initializer=init_ops.zeros_initializer
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        # 提取指定通道
        if y_true.shape.ndims == 4:
            y_true = y_true[:, :, :, self.channel_id]  # 'channel last'
        elif y_true.shape.ndims == 3:
            y_true = y_true[:, :, self.channel_id]
        else:
            pass

        if y_pred.shape.ndims == 4:
            y_pred = y_pred[:, :, :, self.channel_id]
        elif y_pred.shape.ndims == 3:
            y_pred = y_pred[:, :, self.channel_id]
        else:
            pass

        # 预测结果二值化
        y_pred = tf.math.greater(y_pred, self.thresh)

        # 计算混淆矩阵
        cm = tf.math.confusion_matrix(
            tf.reshape(y_true, [-1]),
            tf.reshape(y_pred, [-1]),
            num_classes=2,
            dtype=self._dtype
        )
        return self.class_cm.assign_add(cm)

    def result(self):
        # 计算TP、FP、FN
        tp = math_ops.cast(self.class_cm[1, 1], dtype=self._dtype)
        fp = math_ops.cast(self.class_cm[0, 1], dtype=self._dtype)
        fn = math_ops.cast(self.class_cm[1, 0], dtype=self._dtype)

        denominator = tp + fp + fn

        # 更新IOU
        num_valid_entries = math_ops.reduce_sum(
            math_ops.cast(math_ops.not_equal(denominator, 0), dtype=self._dtype)
        )
        iou = math_ops.div_no_nan(tp, denominator)

        return math_ops.div_no_nan(math_ops.reduce_sum(iou, name=self.name), num_valid_entries)

    def reset_state(self):
        backend.set_value(self.class_cm, np.zeros((2, 2)))


class AiIOU(BinaryIOU):

    def __init__(self, name='ai_iou', channel_id=1, thresh=0.5, **kwargs):
        super(AiIOU, self).__init__(name, channel_id, thresh, **kwargs)


class ArIOU(BinaryIOU):

    def __init__(self, name='ar_iou', channel_id=2, thresh=0.5, **kwargs):
        super(ArIOU, self).__init__(name, channel_id, thresh, **kwargs)


class BgIOU(BinaryIOU):

    def __init__(self, name='bg_iou', channel_id=0, thresh=0.5, **kwargs):
        super(BgIOU, self).__init__(name, channel_id, thresh, **kwargs)


class NeglectIOU(BinaryIOU):

    def __init__(self, name='neglect_iou', channel_id=3, thresh=0.5, **kwargs):
        super(NeglectIOU, self).__init__(name, channel_id, thresh, **kwargs)


class OthersIOU(BinaryIOU):

    def __init__(self, name='others_iou', channel_id=4, thresh=0.5, **kwargs):
        super(OthersIOU, self).__init__(name, channel_id, thresh, **kwargs)


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

    # m = tf.keras.metrics.TruePositives()
    # m.update_state(gt[0], pd[0])
    # print(m.result().numpy())

    m = BinaryIOU('iou', 0)
    m.update_state(gt, pd)
    print(m.result().numpy())
