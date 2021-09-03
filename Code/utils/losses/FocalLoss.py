"""
@Date: 2021/9/3 上午10:52
@Author: Chen Zhang
@Brief: Focal loss implement with tf2
"""
import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.keras.losses import Loss


class FocalLoss(Loss):
    """
    focal loss:
        正样本 y = 1:
            loss = - alpha * (1 - y') ^ gamma * log(y')
        负样本 y = 0:
            loss = - (1 - alpha) * y' ^ gamma * log(1 - y')
        alpha: 平衡因子，可以平衡正负样本的重要性。
        gamma: 调节因子，改变对简单样本和困难样本的关注度。

        当gamma大于1时，简单样本的预测值y'会接近1，而(1-y')^gamma会降低简单那样本在损失中的权重；
        困难样本的预测值y'接近0，(1-y')^gamma会增加困难样本在损失中的权重。

        当alpha=0.5时，正负样本均衡；当alpha<0.5时正样本多于负样本；当alpha>0.5时，正样本少于负样本
    """
    def call(self, y_true, y_pred, alpha=0.5, gamma=2):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        focal_loss = y_true * (-alpha * math_ops.pow(1 - y_pred, gamma) * math_ops.log(y_pred)) + (1 - y_true) * (
                - (1 - alpha) * math_ops.pow(y_pred, gamma) * math_ops.log(1 - y_pred))
        # 这里直接返回focal_loss的话，tensorflow会自动求均值，可能是父类Loss的原因
        return focal_loss


if __name__ == '__main__':
    true = [1.0, 1.0, 0.0, 0.0]
    pred = [0.8, 0.4, 0.7, 0.1]
    print(FocalLoss()(true, pred))  # 0.116223834
