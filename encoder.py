import tensorflow as tf
from tensorflow import keras
from multiHeadAttention import MultiHeadAttention

# Encoder Layer

class TransformerEncoder(keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        # Layer Normalization
        self.layer_norm_1 = keras.layers.LayerNormalization()
        self.layer_norm_2 = keras.layers.LayerNormalization()
        
        # Multi-Head Attention
        self.global_self_attention = MultiHeadAttention(embed_dim=embed_dim, h=num_heads)
        
        # Feed Forward Network
        self.feed_forward = keras.Sequential(
            [keras.layers.Dense(dense_dim, activation="relu"),
             keras.layers.Dense(embed_dim),]
        )

    def call(self, x):
        # Self-attention + residual + normalization
        x = self.layer_norm_1(x + self.global_self_attention(q=x, k=x, v=x))
        # Feed-forward + residual + normalization
        x = self.layer_norm_2(x + self.feed_forward(x))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config