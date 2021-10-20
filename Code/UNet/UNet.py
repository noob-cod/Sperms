"""
@Date: 2021/9/1 下午4:40
@Author: Chen Zhang
@Brief:
"""
from typing import Tuple

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, UpSampling2D
from tensorflow.keras.layers import Cropping2D, Concatenate


class UNet:

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 filter_root: int,  # Floor 1卷积核数量
                 depth: int,  # UNet深度（Floor数量）
                 out_dim: int,  # UNet最终输出的通道数
                 activation_type: str = 'relu',  # 默认激活函数
                 kernel_initializer_type: str = 'he_normal',  # 核初始化方式，默认为he_normal
                 dropout: int = 0,  # Dropout值，默认不带有Dropout
                 batch_norm: bool = True  # 是否包含批正则化层
                 ):
        """
        :param filter_root: 第一次池化操作前的卷积核数量
        :param depth: 网络的深度，数值上比池化操作大1
        """
        self.input_shape = input_shape
        self.filter_root = filter_root
        self.depth = depth
        self.out_dim = out_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation_type = activation_type
        self.kern_ini_type = kernel_initializer_type

    def get_model(self, log=False):
        fm_dict = {}  # 保存压缩路径上的特征图

        inputs = Input(shape=self.input_shape)

        x = Conv2D(self.filter_root, [3, 3], kernel_initializer=self.kern_ini_type, padding='same',
                   name='encode_fl1_conv1')(inputs)
        if self.batch_norm:
            x = BatchNormalization(name='encode_fl1_BN1')(x)
        x = Activation(self.activation_type, name='encode_fl1_act1')(x)

        x = Conv2D(self.filter_root, [3, 3], kernel_initializer=self.kern_ini_type, padding='same',
                   name='encode_fl1_conv2')(x)
        if self.batch_norm:
            x = BatchNormalization(name='encode_fl1_BN2')(x)
        x = Activation(self.activation_type, name='encode_fl1_act2')(x)

        fm_dict[0] = x
        if log:
            print('压缩路径上，Floor 1的输出维度为：Batch:{}, Width:{}, Height:{}, Channels:{}'.format(
                x.shape[0], x.shape[1], x.shape[2], x.shape[3]))

        # 压缩路径上，第2个Floor至最后一个Floor
        for floor in range(self.depth-1):
            filter_nums = self.filter_root * 2 ** (floor + 1)  # 当前floor卷积核数量

            # 最后一个Floor之前的所有Floor
            if floor != self.depth - 2:
                x = MaxPooling2D(pool_size=(2, 2), name='encode_fl{}_maxpool'.format(floor+2))(x)
                if self.dropout != 0:
                    x = Dropout(self.dropout, name='encode_fl{}_dropout'.format(floor+2))(x)

                x = Conv2D(filter_nums, [3, 3], kernel_initializer=self.kern_ini_type, padding='same',
                           name='encode_fl{}_conv1'.format(floor+2))(x)
                if self.batch_norm:
                    x = BatchNormalization(name='encode_fl{}_BN1'.format(floor+2))(x)
                x = Activation(self.activation_type, name='encode_fl{}_act1'.format(floor+2))(x)

                x = Conv2D(filter_nums, [3, 3], kernel_initializer=self.kern_ini_type, padding='same',
                           name='encode_fl{}_conv2'.format(floor+2))(x)
                if self.batch_norm:
                    x = BatchNormalization(name='encode_fl{}_BN2'.format(floor+2))(x)
                x = Activation(self.activation_type, name='encode_fl{}_act2'.format(floor+2))(x)

                fm_dict[floor+1] = x

                if log:
                    print('压缩路径上，Floor {}的输出维度为：Batch:{}, Width:{}, Height:{}, Channels:{}'.format(
                        floor+2, x.shape[0], x.shape[1], x.shape[2], x.shape[3]
                    ))

            else:
                x = MaxPooling2D(pool_size=(2, 2), name='encode_fl{}_maxpool'.format(self.depth))(x)
                if self.dropout != 0:
                    x = Dropout(self.dropout, name='encode_fl{}_dropout'.format(self.depth))(x)

                x = Conv2D(filter_nums, [3, 3], kernel_initializer=self.kern_ini_type, padding='same',
                           name='encode_fl{}_conv1'.format(self.depth))(x)
                if self.batch_norm:
                    x = BatchNormalization(name='encode_fl{}_BN1'.format(self.depth))(x)
                x = Activation(self.activation_type, name='encode_fl{}_act1'.format(self.depth))(x)

                x = Conv2D(filter_nums, [3, 3], kernel_initializer=self.kern_ini_type, padding='same',
                           name='encode_fl{}_conv2'.format(self.depth))(x)
                if self.batch_norm:
                    x = BatchNormalization(name='encode_fl{}_BN2'.format(self.depth))(x)
                x = Activation(self.activation_type, name='encode_fl{}_act2'.format(self.depth))(x)

        # 扩展路径上，从倒数第2个Floor至第1个Floor
        for floor in range(self.depth-2, -1, -1):
            filter_nums = self.filter_root * 2 ** floor
            # 利用“上采样+卷积”来模拟“反卷积”，避免棋盘格效应
            # convtp = convolve transpose
            x = UpSampling2D(size=(2, 2), name='decode_fl{}_convtp_upsampling'.format(floor+1))(x)
            x = Conv2D(filter_nums, [3, 3], kernel_initializer=self.kern_ini_type, padding='same',
                       name='decode_fl{}_convtp_conv'.format(floor+1))(x)

            if log:
                print('在扩展路径上，Floor {}的输出维度为：Batch:{}, Width:{}, Height:{}, Channels:{}'.format(
                    floor+1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]
                ))

            # 特征图的裁剪&拼接
            diff_v = fm_dict[floor].shape[1] - x.shape[1]
            diff_h = fm_dict[floor].shape[2] - x.shape[2]
            y = Cropping2D(cropping=((diff_v//2, diff_v//2), (diff_h//2, diff_h//2)),
                           name='decode_fl{}_crop'.format(floor+1))(fm_dict[floor])
            x = Concatenate(axis=-1, name='decode_fl{}_concatenate'.format(floor+1))([x, y])
            if self.dropout != 0:
                x = Dropout(self.dropout, name='decode_fl{}_dropout'.format(floor+1))(x)

            x = Conv2D(filter_nums, [3, 3], kernel_initializer=self.kern_ini_type, padding='same',
                       name='decode_fl{}_conv1'.format(floor+1))(x)
            if self.batch_norm:
                x = BatchNormalization(name='decode_fl{}_BN1'.format(floor+1))(x)
            x = Activation(self.activation_type, name='decode_fl{}_act1'.format(floor+1))(x)

            x = Conv2D(filter_nums, [3, 3], kernel_initializer=self.kern_ini_type, padding='same',
                       name='decode_fl{}_conv2'.format(floor+1))(x)
            if self.batch_norm:
                x = BatchNormalization(name='decode_fl{}_BN2'.format(floor+1))(x)
            x = Activation(self.activation_type, name='decode_fl{}_act2'.format(floor+1))(x)

        outputs = Conv2D(self.out_dim, (1, 1), activation='sigmoid')(x)

        return Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    my_model = UNet((256, 256, 3), 16, 5, 5)
    unet = my_model.get_model()
    unet.build((256, 256, 3))
    unet.summary()
