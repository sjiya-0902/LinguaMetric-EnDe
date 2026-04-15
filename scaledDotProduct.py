import tensorflow as tf
from casualMasking import mask_attn_weights

def scaled_dot_product_attention_with_weights(q, k, v, use_causal_mask=False):
    # q,k,v shape: [batch, heads, seq_len, depth]
    d_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scores = tf.matmul(q, k, transpose_b=True)  # [batch, heads, q_len, k_len]
    scaled_scores = scores / tf.math.sqrt(d_k)
    if use_causal_mask:
        scaled_scores = mask_attn_weights(scaled_scores)
    weights = tf.nn.softmax(scaled_scores, axis=-1)  # attention weights
    output = tf.matmul(weights, v)
    return output, weights