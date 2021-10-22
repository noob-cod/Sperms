"""
@Date: 2021/10/22 下午3:01
@Author: Chen Zhang
@Brief:

注意事项：
    当TransUNet中加入了CNN结构时，Patch Embedding改为对CNN特征图中的1x1尺寸的patch进行，而非对原图中的patch_size x patch_size尺寸
    的patch。
"""
import tensorflow as tf

from typing import List
from CNN import CNN
from TransformerLayer import TransformerLayer


class TransUnet:

    def __init__(self,
                 input_dim: List,
                 out_dim: int,
                 cnn_model: tf.keras.Model,
                 cnn_filter_root: int,
                 tfl_num: int,
                 patch_size: int,
                 d_model: int,
                 num_heads: int,
                 dff: int,
                 dropout_rate: float = 0.8
                 ):
        """
        :param input_dim: 输入图像的维度，(Batch, H, W, C)
        :param out_dim: 输出通道数
        :param cnn_model: CNN特征提取器
        :param cnn_filter_root: CNN第一层的卷积核
        :param tfl_num: Transformer Layer的层数
        :param patch_size: Image Sequentialization后的正方形Patch的尺寸
        :param d_model: Self-attention编码的维度
        :param num_heads: 多头的头数
        :param dff: 点式前馈模块首层的维度
        :param dropout_rate:  可训练Positional Embedding的丢弃率
        """
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.cnn_model = cnn_model
        self.cnn_filter_root = cnn_filter_root
        self.transformer_layer_num = tfl_num
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

    def get_model(self, log=False):
        # 1.Inputs
        inputs = tf.keras.Input(shape=self.input_dim)  # (bs, H, W, C)
        if log:
            print('Shape of src inputs is {}'.format(inputs.shape))

        # 2.CNN backbone
        out, features = self.cnn_model(self.cnn_filter_root)(inputs)  # (bs, h, w, c)
        if log:
            print('Shape of outputs of CNN Block is {}'.format(out.shape))

        # 3.Linear Projection (Reshape + Patch Embedding)
        patch_embedding = tf.keras.layers.Conv2D(self.d_model,
                                                 kernel_size=[self.patch_size, self.patch_size],
                                                 strides=[self.patch_size, self.patch_size])(out)  # 维度有问题！！！

        # (bs, h/patch_size, w/patch_size, d_model)
        # embedding操作逻辑上等价于卷积核大小为d_model，步长为d_model的二维卷积
        bs, h, w, d = patch_embedding.shape
        if log:
            print('Shape of Embedding result is {}'.format(patch_embedding.shape))
        # patch_embedding = tf.reshape(patch_embedding, shape=[bs, h*w, d])  # (bs, n, d), n为patch的数量
        patch_embedding = tf.keras.layers.Reshape([h*w, d])(patch_embedding)
        if log:
            print('Shape of Patch Embedding is {}'.format(patch_embedding.shape))
        pos_embedding = tf.Variable(tf.zeros(shape=[1, h*w, d]), trainable=True)  # (1, n, d)
        out = patch_embedding + pos_embedding

        # 4.Transformer Encoder
        for _ in range(self.transformer_layer_num):
            out = TransformerLayer(self.d_model, self.num_heads, self.dff, self.dropout_rate)(out)  # (bs, n, d)
        if log:
            print('Shape of Transformer outputs is {}'.format(out.shape))

        # 5.Reshape
        bs, _, d = out.shape
        out = tf.keras.layers.Reshape([h, w, d])(out)
        if log:
            print('Transformer outputs is reshaped to {}'.format(out.shape))

        # 6.U-shape Decoder
        out = tf.keras.layers.Conv2D(self.cnn_filter_root*8, kernel_size=[3, 3], padding='same', activation='relu',
                                     kernel_initializer='he_normal')(out)
        if log:
            print('Shape of inputs of Decoder is {}'.format(out.shape))

        for i in range(2, -1, -1):
            out = tf.keras.layers.UpSampling2D(interpolation='bilinear')(out)
            if log:
                print('Shape of features[{}] is {}'.format(i, features[i].shape))
            out = tf.keras.layers.Concatenate()([features[i], out])  # 此处需要注意features的索引值与i的关系
            out = tf.keras.layers.Conv2D(self.cnn_filter_root**i, kernel_size=[3, 3], padding='same',
                                         kernel_initializer='he_normal')(out)
            out = tf.keras.layers.BatchNormalization()(out)
            out = tf.keras.layers.Conv2D(self.cnn_filter_root ** i, kernel_size=[3, 3], padding='same',
                                         kernel_initializer='he_normal')(out)
            out = tf.keras.layers.BatchNormalization()(out)
            if log:
                print('Shape of {}-th floor of decoder is {}'.format(i, out.shape))
        out = tf.keras.layers.UpSampling2D(interpolation='bilinear')(out)
        out = tf.keras.layers.Conv2D(self.cnn_filter_root // 2, kernel_size=[3, 3], padding='same',
                                     kernel_initializer='he_normal')(out)
        out = tf.keras.layers.BatchNormalization()(out)

        # 7.Segmentation head
        outputs = tf.keras.layers.Conv2D(self.out_dim, [1, 1], activation='sigmoid')(out)

        return tf.keras.Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    myModel = TransUnet([512, 512, 3], 5, CNN, 32, 12, 2, 20, 10, 20)
    model = myModel.get_model(log=True)
    model.summary()
