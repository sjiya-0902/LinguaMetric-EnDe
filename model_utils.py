import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import string
import re
import os

# Importing custom layers
from PositionalEmbedding import PositionalEmbedding
from multiHeadAttention import MultiHeadAttention
from encoder import TransformerEncoder
from decoder import TransformerDecoder

# Re-implemented custom standardization to avoid importing tokenizer.py
def custom_standardization(input_string):
    strip_chars = string.punctuation
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", "")

class GermanTranslator:
    def __init__(self, model_path, source_vocab_path, target_vocab_path, max_decoded_sentence_length=30):
        self.model_path = model_path
        self.source_vocab_path = source_vocab_path
        self.target_vocab_path = target_vocab_path
        self.max_decoded_sentence_length = max_decoded_sentence_length
        self.model = None
        self.source_vectorization = None
        self.target_vectorization = None
        self.target_index_lookup = None
        
        self.load_resources()

    def load_resources(self):
        # Load Vocabularies
        with open(self.source_vocab_path, 'r', encoding='utf-8') as f:
            source_vocab = json.load(f)
        with open(self.target_vocab_path, 'r', encoding='utf-8') as f:
            target_vocab = json.load(f)

        # Re-create Vectorizers
        self.source_vectorization = keras.layers.TextVectorization(
            vocabulary=source_vocab,
            output_mode="int",
            output_sequence_length=30,
            standardize=custom_standardization
        )

        self.target_vectorization = keras.layers.TextVectorization(
            vocabulary=target_vocab,
            output_mode="int",
            output_sequence_length=31, # sequence_length + 1
            standardize=custom_standardization
        )

        self.target_index_lookup = dict(zip(range(len(target_vocab)), target_vocab))

        # Load Model
        custom_objects = {
            "PositionalEmbedding": PositionalEmbedding,
            "MultiHeadAttention": MultiHeadAttention,
            "TransformerEncoder": TransformerEncoder,
            "TransformerDecoder": TransformerDecoder,
            "custom_standardization": custom_standardization
        }
        self.model = keras.models.load_model(self.model_path, custom_objects=custom_objects)

    def decode_sequence(self, input_sentence):
        tokenized_input_sentence = self.source_vectorization([input_sentence])
        decoded_sentence = "[start]"
        for i in range(self.max_decoded_sentence_length):
            tokenized_target_sentence = self.target_vectorization(
                [decoded_sentence])[:, :-1]
            predictions = self.model(
                [tokenized_input_sentence, tokenized_target_sentence])
            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = self.target_index_lookup[sampled_token_index]
            decoded_sentence += " " + sampled_token
            if sampled_token == "[end]":
                break
        return decoded_sentence
