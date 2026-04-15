import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import string
import re
import json

# loading the dataset
df = pd.read_csv('english_to_german.csv')
df.head()

# cleaning the dataset
df.columns = df.columns.str.strip()
df['source'] = df['English']
df['target'] = df['German'].apply(lambda x: '[start] ' + x + ' [end]')
df = df.drop(['English', 'German'], axis=1)
display(df.sample(5))

# shuffling the dataset
df = df.sample(frac=1).reset_index(drop=True)

# splitting into train, validation, and test sets
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.2)
test_size = int(len(df) * 0.1)

train_df = df[:train_size]
val_df = df[train_size:train_size+val_size]
test_df = df[train_size+val_size:]

# hyperparameters
max_tokens = 25000
sequence_length = 30

# removing punctuations except "[" and "]"
strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", "")

# source tokenizer
source_vectorization = keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length,
)
# target tokenizer
target_vectorization = keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length + 1, # +1 for the shift
    standardize=custom_standardization, 
)

# adapting the tokenizers to the training data
train_source_texts = train_df['source'].values
train_target_texts = train_df['target'].values
source_vectorization.adapt(train_source_texts)
target_vectorization.adapt(train_target_texts)

# saving the tokenizers
json.dump(source_vectorization.get_vocabulary(), open("source_vocab_re.json","w"))
json.dump(target_vectorization.get_vocabulary(), open("target_vocab_re.json","w"))