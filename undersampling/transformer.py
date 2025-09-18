from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization, Dropout, Dense
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

class PositionWiseFeedForward(Layer):

    def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
        self._model_dim = model_dim
        self._inner_dim = inner_dim
        self._trainable = trainable
        super(PositionWiseFeedForward, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weights_inner = self.add_weight(
            shape=(input_shape[-1], self._inner_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_inner")

        self.weights_out = self.add_weight(
            shape=(self._inner_dim, self._model_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_out")

        self.bais_inner = self.add_weight(
            shape=(self._inner_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_inner")

        self.bais_out = self.add_weight(
            shape=(self._model_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_out")
        super(PositionWiseFeedForward, self).build(input_shape)

    def call(self, inputs):
        if K.dtype(inputs) != 'float32':
            inputs = K.cast(inputs, 'float32')
        inner_out = K.relu(K.dot(inputs, self.weights_inner) + self.bais_inner)
        outputs = K.dot(inner_out, self.weights_out) + self.bais_out
        return outputs

class PositionEncoding(Layer):

    def __init__(self, model_dim, **kwargs):
        self._model_dim = model_dim
        super(PositionEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        seq_length = inputs.shape[1]
        position_encodings = np.zeros((seq_length, self._model_dim))
        for pos in range(seq_length):
            for i in range(self._model_dim):
                position_encodings[pos, i] = pos / np.power(10000, (i - i % 2) / self._model_dim)
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2])  # 2i
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2])  # 2i+1
        position_encodings = K.cast(position_encodings, 'float32')
        return position_encodings

class LayerNormalization(Layer):

    def __init__(self, epsilon=1e-8, **kwargs):
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zero',
            name='beta')
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='one',
            name='gamma')
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
        outputs = self.gamma * normalized + self.beta
        return outputs

class ScaledDotProductAttention(Layer):

    def __init__(self, masking=True, future=False, dropout_rate=0.3, **kwargs):
        self._masking = masking
        self._future = future
        self._dropout_rate = dropout_rate
        self._masking_num = -2 ** 32 + 1
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def mask(self, inputs, masks):
        masks = K.cast(masks, 'float32')
        masks = K.tile(masks, [K.shape(inputs)[0] // K.shape(masks)[0], 1])
        masks = K.expand_dims(masks, 1)
        outputs = inputs + masks * self._masking_num
        return outputs

    def call(self, inputs):
        queries, keys, values = inputs
        matmul = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1]))
        scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5
        softmax_out = K.softmax(scaled_matmul)
        out = K.dropout(softmax_out, self._dropout_rate)
        outputs = K.batch_dot(out, values)
        return outputs

class MultiHeadAttention(Layer):

    def __init__(self, n_heads, head_dim, dropout_rate=0.3, masking=False, future=False, trainable=True, **kwargs):
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._dropout_rate = dropout_rate
        self._masking = masking
        self._future = future
        self._trainable = trainable
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self._weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_queries')
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_keys')
        self._weights_values = self.add_weight(
            shape=(input_shape[2][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_values')
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs):
        queries, keys, values = inputs
        queries_linear = K.dot(queries, self._weights_queries)
        keys_linear = K.dot(keys, self._weights_keys)
        values_linear = K.dot(values, self._weights_values)

        queries_multi_heads = tf.concat(tf.split(queries_linear, self._n_heads, axis=2), axis=0)
        keys_multi_heads = tf.concat(tf.split(keys_linear, self._n_heads, axis=2), axis=0)
        values_multi_heads = tf.concat(tf.split(values_linear, self._n_heads, axis=2), axis=0)

        attention = ScaledDotProductAttention(masking=self._masking, future=self._future, dropout_rate=self._dropout_rate)
        att_out = attention([queries_multi_heads, keys_multi_heads, values_multi_heads])

        outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)
        return outputs

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, n_heads, head_dim, dropout_rate=0.3, **kwargs):
        super(CrossAttention, self).__init__(**kwargs)
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._dropout_rate = dropout_rate

    def build(self, input_shape):
        self._weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            name='weights_queries'
        )
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            name='weights_keys'
        )
        self._weights_values = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            name='weights_values'
        )
        super(CrossAttention, self).build(input_shape)

    def call(self, inputs):
        queries, keys, values = inputs

        # 线性变换
        queries_linear = tf.matmul(queries, self._weights_queries)
        keys_linear = tf.matmul(keys, self._weights_keys)
        values_linear = tf.matmul(values, self._weights_values)

        # 分头处理
        queries_multi_heads = tf.concat(tf.split(queries_linear, self._n_heads, axis=2), axis=0)
        keys_multi_heads = tf.concat(tf.split(keys_linear, self._n_heads, axis=2), axis=0)
        values_multi_heads = tf.concat(tf.split(values_linear, self._n_heads, axis=2), axis=0)

        # 计算注意力
        attention = ScaledDotProductAttention(dropout_rate=self._dropout_rate)
        att_out = attention([queries_multi_heads, keys_multi_heads, values_multi_heads])

        # 合并头
        outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)
        return outputs

class Transformer_Merged_CrossAttention(tf.keras.layers.Layer):
    def __init__(self, model_dim=72, n_heads=8, encoder_stack=3, feed_forward_size=256, dropout_rate=0.5, **kwargs):
        super(Transformer_Merged_CrossAttention, self).__init__(**kwargs)
        self._encoder_stack = encoder_stack
        self._model_dim = model_dim
        self._n_heads = n_heads
        self._feed_forward_size = feed_forward_size
        self._dropout_rate = dropout_rate

        # 组件初始化
        self.positionencoding = PositionEncoding(self._model_dim)
        self.feedforward = PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
        self.crossattention = CrossAttention(self._n_heads, self._model_dim // self._n_heads)

        self.dropout = Dropout(self._dropout_rate)
        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()

    def call(self, inputs):
        enhancer_embeddings, promoter_embeddings = inputs
       
        enhancer_encodings = enhancer_embeddings + self.positionencoding(enhancer_embeddings)
        promoter_encodings = promoter_embeddings + self.positionencoding(promoter_embeddings)

        enhancer_encodings = self.dropout(enhancer_encodings)
        promoter_encodings = self.dropout(promoter_encodings)

        for _ in range(self._encoder_stack):
            enhancer_to_promoter = self.crossattention([enhancer_encodings, promoter_encodings, promoter_encodings])
            promoter_to_enhancer = self.crossattention([promoter_encodings, enhancer_encodings, enhancer_encodings])

            # 残差连接 + 归一化
            enhancer_encodings = self.layer_norm_1(enhancer_encodings + enhancer_to_promoter)
            enhancer_encodings = self.layer_norm_2(enhancer_encodings + self.feedforward(enhancer_encodings))

            promoter_encodings = self.layer_norm_1(promoter_encodings + promoter_to_enhancer)
            promoter_encodings = self.layer_norm_2(promoter_encodings + self.feedforward(promoter_encodings))

        return enhancer_encodings, promoter_encodings
