from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Concatenate, Dense, Conv1D, Dropout, BatchNormalization, MaxPooling1D
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from transformer import Transformer_Merged_CrossAttention  

def build_model(en_max_len, pr_max_len, onehot_dim=27, embed_dim=64, num_heads=8, feed_forward_size=128):
    # 输入层（one-hot 编码的序列）
    enhancer_input = Input(shape=(en_max_len, onehot_dim), dtype=tf.float32, name="enhancer_input")
    promoter_input = Input(shape=(pr_max_len, onehot_dim), dtype=tf.float32, name="promoter_input")

    def conv_block(input_layer):
        x = Conv1D(embed_dim * 2, kernel_size=5, activation="relu", kernel_regularizer=l2(1e-4))(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = MaxPooling1D(pool_size=2)(x) 

        x = Conv1D(embed_dim, kernel_size=3, activation="relu", kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        return x

    enhancer_conv = conv_block(enhancer_input)
    promoter_conv = conv_block(promoter_input)

    transformer_encoder = Transformer_Merged_CrossAttention(
        model_dim=embed_dim, n_heads=num_heads, feed_forward_size=feed_forward_size, encoder_stack=3
    )
    enhancer_encoded, promoter_encoded = transformer_encoder([enhancer_conv, promoter_conv])

    # **全局池化**
    enhancer_pooled = GlobalAveragePooling1D()(enhancer_encoded)
    promoter_pooled = GlobalAveragePooling1D()(promoter_encoded)

    # **拼接最终特征**
    merged_features = Concatenate()([enhancer_pooled, promoter_pooled])
    merged_features = Dropout(0.2)(merged_features)

    # **全连接层**
    dense = Dense(74, activation="relu", kernel_regularizer=l2(1e-4))(merged_features)
    dense = Dropout(0.2)(dense)
    output = Dense(1, activation="sigmoid")(dense) 


    model = tf.keras.Model(inputs=[enhancer_input, promoter_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="AUROC", curve="ROC"),
            tf.keras.metrics.AUC(name="AUPRC", curve="PR"),
            tf.keras.metrics.Precision(name="Precision"),
            tf.keras.metrics.Recall(name="Recall"),
            tf.keras.metrics.F1Score(name="F1", threshold=0.5)  
        ]
    )
    return model
