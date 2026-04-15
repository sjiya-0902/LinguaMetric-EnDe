import tensorflow as tf
from tensorflow import keras
from multiHeadAttention import MultiHeadAttention

# Decoder Layer

class TransformerDecoder(keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        # Self-attention but with causal mask (can’t see future tokens)
        self.causal_self_attention = MultiHeadAttention(embed_dim=embed_dim, h=num_heads)
        
        # Cross-attention to encoder output
        self.cross_attention = MultiHeadAttention(embed_dim=embed_dim, h=num_heads)
        
        # Feed Forward Network
        self.feed_forward = keras.Sequential(
            [keras.layers.Dense(dense_dim, activation="relu"),
             keras.layers.Dense(embed_dim),]
        )

        # Layer Normalization
        self.layer_norm_1 = keras.layers.LayerNormalization()
        self.layer_norm_2 = keras.layers.LayerNormalization()
        self.layer_norm_3 = keras.layers.LayerNormalization()

    # context = encoder output
    def call(self, x, context, return_cross_attn=False):
        # Masked self-attention + residual + norm
        x = self.layer_norm_1(x + self.causal_self_attention(q=x, k=x, v=x, use_causal_mask=True))
        
        # get cross-attention outputs and weights
        cross_out = self.cross_attention(q=x, k=context, v=context, return_attn_weights=return_cross_attn)
        if return_cross_attn:
            cross_concat, attn_weights = cross_out
            x = self.layer_norm_2(x + cross_concat)
        else:
            x = self.layer_norm_2(x + cross_out)
            attn_weights = None

        # FFN + residual + norm
        x = self.layer_norm_3(x + self.feed_forward(x))
        if return_cross_attn:
            return x, attn_weights
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config