"""
@Date: 2021/9/1 下午4:50
@Author: Chen Zhang
@Brief: UNetpp
"""
from typing import Tuple

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, UpSampling2D
from tensorflow.keras.layers import Concatenate


class Conv2DTranspose(Model):

    def __init__(self,
                 filter_root: [int],
                 floor_index: [int],
                 kernel_initializer_type: [str] = 'he_normal',
                 ):
        super(Conv2DTranspose, self).__init__()
        self.filter_root = filter_root
        self.floor_index = floor_index
        self.filter_num = self.filter_root * 2 ** self.floor_index

        self.upsample2d = UpSampling2D(size=(2, 2))
        self.conv2d = Conv2D(self.filter_num, [3, 3], kernel_initializer=kernel_initializer_type, padding='same')

    def call(self, inputs, training=None, mask=None):
        x = self.upsample2d(inputs)
        return self.conv2d(x)  # 是否需要激活函数？？？


class UNetPPNeuron(Model):

    def __init__(self,
                 filter_root: [int],  # 该单元所在的整个网络中，第一个Floor中卷积核的数量
                 floor_index: [int],
                 inner_index: [int],
                 activation_type: [str] = 'relu',
                 kernel_initializer_type: [str] = 'he_normal',
                 batch_norm: [bool] = True
                 ):
        super(UNetPPNeuron, self).__init__()
        self.filter_root = filter_root
        self.batch_norm = batch_norm

        self.floor_index = floor_index  # 该单元所在的层序号，从0开始计算
        self.inner_index = inner_index  # 该单元在层内的序号，从0开始计算

        self.filter_num = self.filter_root * 2 ** self.floor_index  # 当前单元实际的卷积核数量

        self.conv2d_1 = Conv2D(self.filter_num, [3, 3], kernel_initializer=kernel_initializer_type, padding='same')
        self.conv2d_2 = Conv2D(self.filter_num, [3, 3], kernel_initializer=kernel_initializer_type, padding='same')
        self.bn = BatchNormalization()
        self.activation = Activation(activation_type)

    def call(self, inputs, training=None, mask=None):
        x = self.conv2d_1(inputs)
        if self.batch_norm:
            x = self.bn(x)
        x = self.activation(x)

        x = self.conv2d_2(x)
        if self.batch_norm:
            x = self.bn(x)
        return self.activation(x)


class UNetPP:

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 filter_root: [int],
                 depth: [int],
                 out_dim: [int],
                 activation_type: [str] = 'relu',
                 kernel_initializer_type: [str] = 'he_normal',
                 batch_norm: [bool] = True
                 ):
        self.input_shape = input_shape
        self.filter_root = filter_root
        self.depth = depth
        self.out_dim = out_dim
        self.activation_type = activation_type
        self.kern_ini_type = kernel_initializer_type
        self.batch_norm = batch_norm

    def get_model(self):
        inputs = Input(shape=self.input_shape)

        neurons = [[None] * self.depth for _ in range(self.depth)]  # 存放不同位置单元的输出

        neurons[0][0] = UNetPPNeuron(self.filter_root, 0, 0, activation_type=self.activation_type,
                                     kernel_initializer_type=self.kern_ini_type, batch_norm=self.batch_norm)(inputs)

        floor = 0

        while floor != self.depth-1:

            floor += 1  # 层数

            for inner_ind in range(floor+1):
                floor_ind = floor - inner_ind
                if inner_ind == 0:  # 每个floor的首个单元的输入只有1个，即为上一个floor的首个单元的Maxpooling结果
                    tmp_input = MaxPooling2D(pool_size=(2, 2))(neurons[floor_ind-1][0])
                    neurons[floor_ind][inner_ind] = UNetPPNeuron(self.filter_root, floor_ind, inner_ind,
                                                                 activation_type=self.activation_type,
                                                                 kernel_initializer_type=self.kern_ini_type,
                                                                 batch_norm=self.batch_norm)(tmp_input)
                else:  # 其他单元的输入有2个，即下一个floor中inner_ind-1位置单元上采样结果与同一个floor中前面所有单元结果的concatenate
                    x = Conv2DTranspose(self.filter_root, floor_ind)(neurons[floor_ind+1][inner_ind-1])
                    tmp_input = neurons[floor_ind][:inner_ind]  # 提取同一floor中前面所有单元的结果
                    tmp_input.extend([x])  # 将上采样的结果放在同一list中
                    tmp_input = Concatenate(axis=-1)(tmp_input)  # 合并
                    neurons[floor_ind][inner_ind] = UNetPPNeuron(self.filter_root, floor_ind, inner_ind,
                                                                 activation_type=self.activation_type,
                                                                 kernel_initializer_type=self.kern_ini_type,
                                                                 batch_norm=self.batch_norm)(tmp_input)

        outputs = Conv2D(self.out_dim, (1, 1), activation='sigmoid')(neurons[0][self.depth-1])

        return Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    my_model = UNetPP((400, 400, 3), 16, 5, 2)
    unetpp = my_model.get_model()
    unetpp.build((400, 400, 3))
    unetpp.summary()
