"""
@Date: 2021/10/22 下午2:20
@Author: Chen Zhang
@Brief: Transformer Layer (部分代码来自Google官方）
"""
import keras.layers
import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    """
    计算注意力权重。
    q, k, v 必须具有匹配的前置维度。
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    虽然mask根据其类型有不同的形状，但是mask必须能进行广播转换以便求和。

    :param q: query向量, q = Wq * x，x为输入单词的Embedding结果
    :param k: key向量, k = Wk * x
    :param v: value向量, v = Wv * x
    :param mask: float张量，其形状能转换为(..., seq_len_q, seq_len_k)

    :return: (Attention值, 注意力权重)
    """
    # 1.计算相似程度f(Q, K)
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # 缩放matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 将mask加入到缩放的张量上
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 2.计算attention权重alpha
    # softmax在最后一个轴（seq_len_k）上归一化，因此分数相加等于1
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    # 3.计算attention值
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads):
        """
        多头Attention层

        :param d_model: 模型对词向量编码的维度
        :param num_heads: 头的数量
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        # 多头的W矩阵，len(Wq) = len(Wq_0) + len(Wq_1) + ... + len(Wq_n)
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        若x维度为 b x k x n 维，则将其拆分为 b x k x l x r 维，其中 n = l x r
        拆分最后一个维度到(num_heads, depth)。转置结果使得形状为(batch_size, num_heads, seq_len, depth)

        :param x: 输入向量
        :param batch_size: 批尺寸

        :return: 拆出来的向量
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """
        此处的v, k, q为多头的输入
        """
        batch_size = tf.shape(q)[0]

        # 1.线性层并拆分成多头
        # 多头合并为一个矩阵进行计算，根据输入值，计算q, k, v的合集
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # 将最后一个维度拆分，本质为将每个头输出的q, k, v分别拆分出来
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # 2.按比例缩放的点积注意力
        # 计算每个头的注意力输出
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # 3.多头级联
        # 将scaled_attention转置为self.split_heads()方法分拆之前的格式
        # scaled_attention.shape == (batch_size, seq_len_q, nums_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # 将每个头的attention再拼接起来
        # scaled_attention.shape == (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # 4.最后一层线性层
        # 拼接起来的长向量乘以一个矩阵（tf.keras.layers.Dense是一个可训练的矩阵），就得到了一个短向量。
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_netword(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff), 此处dff为某个维度
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_netword(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None, mask=None):

        out_ln = self.layernorm1(inputs)
        attn_output, _ = self.mha(out_ln, out_ln, out_ln, mask)  # v, k, q
        attn_output = self.dropout1(attn_output,  training=training)
        out1 = inputs + attn_output

        out_ln = self.layernorm2(out1)
        ffn_out = self.ffn(out_ln)
        ffn_out = self.dropout2(ffn_out, training=training)
        out2 = out1 + ffn_out

        return out2


if __name__ == '__main__':
    pass
