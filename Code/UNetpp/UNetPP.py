"""
@Date: 2021/10/18 下午3:34
@Author: Chen Zhang
@Brief: https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py
"""
from typing import Tuple

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Concatenate


class UNetPPNeuron(Model):

    def __init__(self,
                 filter_root: [int],
                 activation_type: str = 'relu',
                 kernel_initializer_type: [str] = 'he_normal',
                 ):
        super().__init__()
        self.conv1 = Conv2D(filter_root, [3, 3], kernel_initializer=kernel_initializer_type, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filter_root, [3, 3], kernel_initializer=kernel_initializer_type, padding='same')
        self.bn2 = BatchNormalization()
        self.relu = Activation(activation_type)

    def call(self, inputs, training=None, mask=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNetPP:

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 filter_root: [int],
                 depth: [int],
                 out_dim: [int],
                 activation_type: [str] = 'relu',
                 kernel_initializer_type: [str] = 'he_normal',
                 batch_norm: [bool] = True,
                 deep_supervision: [bool] = False
                 ):

        self.n_filter = [16, 32, 64, 128, 256]

        self.input_shape = input_shape
        self.filter_root = filter_root
        self.depth = depth
        self.out_dim = out_dim
        self.activation_type = activation_type
        self.kern_ini_type = kernel_initializer_type
        self.batch_norm = batch_norm
        self.deep_supervision = deep_supervision

        self.pool = MaxPooling2D(pool_size=(2, 2))
        self.up = UpSampling2D(size=[2, 2], interpolation='bilinear')

        self.conv0_0 = UNetPPNeuron(self.n_filter[0])
        self.conv1_0 = UNetPPNeuron(self.n_filter[1])
        self.conv2_0 = UNetPPNeuron(self.n_filter[2])
        self.conv3_0 = UNetPPNeuron(self.n_filter[3])
        self.conv4_0 = UNetPPNeuron(self.n_filter[4])

        self.conv0_1 = UNetPPNeuron(self.n_filter[0])
        self.conv1_1 = UNetPPNeuron(self.n_filter[1])
        self.conv2_1 = UNetPPNeuron(self.n_filter[2])
        self.conv3_1 = UNetPPNeuron(self.n_filter[3])

        self.conv0_2 = UNetPPNeuron(self.n_filter[0])
        self.conv1_2 = UNetPPNeuron(self.n_filter[1])
        self.conv2_2 = UNetPPNeuron(self.n_filter[2])

        self.conv0_3 = UNetPPNeuron(self.n_filter[0])
        self.conv1_3 = UNetPPNeuron(self.n_filter[1])

        self.conv0_4 = UNetPPNeuron(self.n_filter[0])

        if self.deep_supervision:
            self.final1 = Conv2D(self.out_dim, [1, 1], activation='sigmoid')
            self.final2 = Conv2D(self.out_dim, [1, 1], activation='sigmoid')
            self.final3 = Conv2D(self.out_dim, [1, 1], activation='sigmoid')
            self.final4 = Conv2D(self.out_dim, [1, 1], activation='sigmoid')
        else:
            self.final = Conv2D(self.out_dim, [1, 1], activation='sigmoid')

    def get_model(self):
        inputs = Input(shape=self.input_shape)

        x0_0 = self.conv0_0(inputs)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(Concatenate(axis=-1)([x0_0, self.up(x1_0)]))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(Concatenate(axis=-1)([x1_0, self.up(x2_0)]))
        x0_2 = self.conv0_2(Concatenate(axis=-1)([x0_0, x0_1, self.up(x1_1)]))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(Concatenate(axis=-1)([x2_0, self.up(x3_0)]))
        x1_2 = self.conv1_2(Concatenate(axis=-1)([x1_0, x1_1, self.up(x2_1)]))
        x0_3 = self.conv0_3(Concatenate(axis=-1)([x0_0, x0_1, x0_2, self.up(x1_2)]))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(Concatenate(axis=-1)([x3_0, self.up(x4_0)]))
        x2_2 = self.conv2_2(Concatenate(axis=-1)([x2_0, x2_1, self.up(x3_1)]))
        x1_3 = self.conv1_3(Concatenate(axis=-1)([x1_0, x1_1, x1_2, self.up(x2_2)]))
        x0_4 = self.conv0_4(Concatenate(axis=-1)([x0_0, x0_2, x0_3, self.up(x1_3)]))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return Model(inputs=inputs, outputs=[output1, output2, output3, output4])
        else:
            output = self.final(x0_4)
            return Model(inputs=inputs, outputs=output)


if __name__ == '__main__':
    my_model = UNetPP((400, 400, 3), 16, 5, 2, deep_supervision=True)
    unetpp = my_model.get_model()
    unetpp.build((400, 400, 3))
    unetpp.summary()
