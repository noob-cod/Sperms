"""
@Date: 2021/10/22 下午1:54
@Author: Chen Zhang
@Brief:  CNN Model
"""
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation


class CNN(Model):

    def __init__(self,
                 filter_root: int,
                 activation_type: str = 'relu',
                 kernel_initializer_type: str = 'he_normal',
                 ):
        super().__init__(self)
        self.conv1 = Conv2D(filters=filter_root, kernel_size=[3, 3], padding='same',
                            kernel_initializer=kernel_initializer_type)
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPooling2D()

        self.conv2 = Conv2D(filters=filter_root*2, kernel_size=[3, 3], padding='same',
                            kernel_initializer=kernel_initializer_type)
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPooling2D()

        self.conv3 = Conv2D(filters=filter_root*4, kernel_size=[3, 3], padding='same',
                            kernel_initializer=kernel_initializer_type)
        self.bn3 = BatchNormalization()
        self.pool3 = MaxPooling2D()

        self.relu = Activation(activation=activation_type)

    def call(self, inputs, train=None, mask=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.pool1(out)
        feature1 = out

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.pool2(out)
        feature2 = out

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.pool3(out)
        feature3 = out

        return out, [feature1, feature2, feature3]


if __name__ == '__main__':
    from tensorflow.keras.layers import Input, Softmax
    inputs = Input(shape=(400, 400, 3))
    x = CNN(filter_root=8)(inputs)
    outputs = Activation('sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
