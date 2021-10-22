"""
@Date: 2021/10/22 下午3:48
@Author: Chen Zhang
@Brief: Patch Embedding Layer
"""
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[4, 4, 128]),
    tf.keras.layers.UpSampling2D(size=(1, 1))
])

model.summary()
