import tensorflow as tf
from tensorflow import keras

# Positional Embedding Layer

class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        # Token embedding: turns word indices into vectors
        self.token_embeddings = keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        # Position embedding: turns position indices into vectors
        self.position_embeddings = keras.layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        embedded_tokens = self.token_embeddings(inputs)
        length = tf.shape(inputs)[-1] # Find length of input sequence
        positions = tf.range(start=0, limit=length, delta=1) # Generate positions (0 to length-1)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
    
    # Mask padding tokens (value 0)
    def compute_mask(self, inputs, mask=None):
        return keras.ops.not_equal(inputs, 0)

    # To save model config later
    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
        })
        return config