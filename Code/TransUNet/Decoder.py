"""
@Date: 2021/10/25 上午10:01
@Author: Chen Zhang
@Brief: TransUNet Decoder

Bug:
    TypeError: Could not build a TypeSpec for <KerasTensor: shape=(None, 128, 128, 64) dtype=float32 (created by layer 'activation_4')> with type KerasTensor

"""
import tensorflow as tf

from typing import List


class DecoderBlock(tf.keras.Model):

    def __init__(self,
                 filter_num: int,
                 kernel_size: List,
                 activation_type: str = 'relu',
                 kernel_initializer: str = 'he_normal',
                 feature=None
                 ):
        """
        :param feature: 需要融合的CNN的特征图，需要与当前分辨率匹配
        :param filter_num: 卷积核数量
        :param kernel_size: 卷积核尺寸
        :param activation_type: 激活函数
        :param kernel_initializer: 卷积和初始化方式
        """
        super(DecoderBlock, self).__init__()
        self.feature = feature

        self.upsample = tf.keras.layers.UpSampling2D(interpolation='bilinear')
        self.concat = tf.keras.layers.Concatenate()

        self.conv1 = tf.keras.layers.Conv2D(filter_num, kernel_size, padding='same',
                                            kernel_initializer=kernel_initializer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation(activation_type)

        self.conv2 = tf.keras.layers.Conv2D(filter_num, kernel_size, padding='same',
                                            kernel_initializer=kernel_initializer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation(activation_type)

    def call(self, inputs, training=None, mask=None):
        print('input:', inputs)
        out = self.upsample(inputs)

        if self.feature is not None:
            out = self.concat([self.feature, out])  # 此处需要注意features的索引值与i的关系

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        print('out:', out)

        return out


class Decoder(tf.keras.Model):

    def __init__(self,
                 block_num: int,
                 filter_num: int,
                 kernel_size: int or List[int, int] = [3, 3],
                 activation_type: str = 'relu',
                 kernel_initializer: str = 'he_normal',
                 features=None,
                 ):
        """
        :param block_num: 要堆积的DecoderBlock的数量
        :param filter_num: 最底层block的卷积核数量
        :param kernel_size: 卷积核尺寸
        :param activation_type: 激活函数类型
        :param kernel_initializer: 核初始化方式
        :param features: 由CNN模块传递的不同分辨率的特征图，按分辨率从小到大排列
        """
        super(Decoder, self).__init__()
        self.features = features

        if self.features:
            self.decoder_block1 = DecoderBlock(filter_num=filter_num, kernel_size=kernel_size,
                                               activation_type=activation_type, kernel_initializer=kernel_initializer,
                                               feature=self.features[0])
            self.decoder_block2 = DecoderBlock(filter_num=filter_num * 2, kernel_size=kernel_size,
                                               activation_type=activation_type, kernel_initializer=kernel_initializer,
                                               feature=self.features[1])
            self.decoder_block3 = DecoderBlock(filter_num=filter_num * 4, kernel_size=kernel_size,
                                               activation_type=activation_type, kernel_initializer=kernel_initializer,
                                               feature=self.features[2])
        else:
            self.decoder_block1 = DecoderBlock(filter_num=filter_num, kernel_size=kernel_size,
                                               activation_type=activation_type, kernel_initializer=kernel_initializer)
            self.decoder_block2 = DecoderBlock(filter_num=filter_num * 2, kernel_size=kernel_size,
                                               activation_type=activation_type, kernel_initializer=kernel_initializer)
            self.decoder_block3 = DecoderBlock(filter_num=filter_num * 4, kernel_size=kernel_size,
                                               activation_type=activation_type, kernel_initializer=kernel_initializer)

    def call(self, inputs, training=None, mask=None):

        out = self.decoder_block1(inputs)
        out = self.decoder_block2(out)
        out = self.decoder_block3(out)

        return out


if __name__ == '__main__':
    model = Decoder(block_num=3, filter_num=16, kernel_size=[3, 3], features=None)
    model.build([1, 512, 512, 3])
    model.summary()
