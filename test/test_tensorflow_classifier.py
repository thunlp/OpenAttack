import tensorflow as tf 
import numpy as np
import pickle
from TAADToolbox.classifiers.tensorflow_classifier import TensorflowClassifier
import nltk

model = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300)),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])



embedding_matrix = np.load('embeddings_glove_50000.npy').transpose(1, 0)
with open('imdb.vocab', 'r') as f:
    vocab_words = f.read().split('\n')
    vocab = dict([(w, i) for i, w in enumerate(vocab_words)])
classifier =  TensorflowClassifier(model, vocab=vocab, max_len=26, embedding=embedding_matrix)
print(classifier.get_pred(["i like apples", "i like apples"]))
print(classifier.get_prob(["i like apples", "i like apples"]))
print(classifier.get_grad(["i like apples", "i like apples"], [1, 1]))