"""
@Date: 2021/11/23 下午4:14
@Author: Chen Zhang
@Brief: 多类Dice的实现

Jaccard Dice = (2 * True * Pred) / (True**2 + Pred**2)
Sorensen Dice = (2 * True * Pred) / (True + Pred)

Dice与IOU的区别：
    1、IOU计算的是预测正确区域的像素数量（TP）占真实区域和预测区域并集的比例。
    2、Dice计算的是预测正确区域的像素的概率的和（Intersection）占
        (真实区域的权重和 + 预测区域的概率和）的比例。
    3、IOU会按阈值将预测值0~1之间的概率二值化，忽略阳性概率小于阈值的位置的影响；
        Dice则会综合考虑所有预测概率大于0的位置的影响。
    4、IOU在模型预测的精确度、预测的边缘等方面比DICE更敏感；Dice更能反应模型的
        综合预测能力。
"""
import tensorflow as tf

from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops


class Dice(tf.keras.metrics.MeanMetricWrapper):

    def __init__(self, name, channel_id, reduction=metrics_utils.Reduction.WEIGHTED_MEAN, smooth=1e-5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.channel_id = channel_id  # 要计算iou的通道，每个通道包含一个类的预测结果
        self.reduction = reduction
        self.smooth = smooth

        self.dice = self.add_weight(name='dice', initializer=init_ops.zeros_initializer)
        if reduction in [metrics_utils.Reduction.SUM_OVER_BATCH_SIZE,
                         metrics_utils.Reduction.WEIGHTED_MEAN]:
            self.count = self.add_weight('count', initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=self._dtype)
        y_pred = tf.cast(y_pred, dtype=self._dtype)

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

        intersection = math_ops.cast(tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0, 1]), dtype=self._dtype)
        union = math_ops.cast(tf.reduce_sum(tf.multiply(y_true, y_true)) + tf.reduce_sum(tf.multiply(y_pred, y_pred)),
                              dtype=self._dtype)

        dice = math_ops.cast(math_ops.div_no_nan(2.0 * intersection + self.smooth, union + self.smooth), dtype=self._dtype)
        dice_sum = math_ops.reduce_sum(dice)

        update_total_dice = self.dice.assign_add(dice_sum)

        if self.reduction == metrics_utils.Reduction.SUM:
            return update_total_dice

        if self.reduction == metrics_utils.Reduction.SUM_OVER_BATCH_SIZE:
            num = math_ops.cast(array_ops.size(dice), dtype=self._dtype)
        elif self.reduction == metrics_utils.Reduction.WEIGHTED_MEAN:
            if sample_weight is None:
                num = math_ops.cast(array_ops.size(dice), dtype=self._dtype)
            else:
                num = math_ops.reduce_sum(sample_weight)
        else:
            raise NotImplementedError('reduction [%s] not implemented' % self.reduction)

        return self.count.assign_add(num)

    def result(self):
        if self.reduction == metrics_utils.Reduction.SUM:
            return array_ops.identity(self.dice)
        elif self.reduction in[
            metrics_utils.Reduction.WEIGHTED_MEAN,
            metrics_utils.Reduction.SUM_OVER_BATCH_SIZE
        ]:
            return math_ops.div_no_nan(self.dice, self.count)
        else:
            raise NotImplementedError('reduction [%s] not implemented' % self.reduction)


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

    m = Dice('Dice', 0, reduction=tf.keras.metrics.Reduction)
    m.update_state(gt, pd)
    print(m.result().numpy())
