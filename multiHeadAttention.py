import tensorflow as tf
from tensorflow import keras
from scaledDotProduct import scaled_dot_product_attention_with_weights

# Multi Head Attention Layer

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, embed_dim, h, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.h = h
        if embed_dim % h != 0:
            raise ValueError("embed_dim must be divisible by number of heads")
        # Linear layers to get Q, K, V vectors
        self.q_linear = keras.layers.Dense(embed_dim)
        self.k_linear = keras.layers.Dense(embed_dim)
        self.v_linear = keras.layers.Dense(embed_dim)
        self.concat_linear = keras.layers.Dense(embed_dim)

    # Split the embedding into h heads
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, shape=(batch_size, -1, self.h, self.embed_dim // self.h))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # Concatenate the heads back into a single embedding
    def concat_heads(self, x, batch_size):
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, self.embed_dim))

    def call(self, q, k, v, use_causal_mask=False, return_attn_weights=False):
        batch_size = tf.shape(k)[0]
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
         # Split heads for multi-head attention
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        out, weights = scaled_dot_product_attention_with_weights(q, k, v, use_causal_mask)
        concat = self.concat_heads(out, batch_size)
        concat = self.concat_linear(concat)
        if return_attn_weights:
            return concat, weights
        return concat

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({"embed_dim": self.embed_dim, "h": self.h})
        return config