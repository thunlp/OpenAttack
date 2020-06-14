import tensorflow as tf 
import numpy as np
import pickle
import TAADToolbox as tat
import nltk

net = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300)),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])



punc = [',', '.', '?', '!']
embedding_matrix = np.random.randn(20, 10)
vocab = dict()
num = 0
for p in punc:
    vocab[p] = num
    num += 1
for w in "i like apples".split():
    vocab[w] = num
    num += 1
classifier =  tat.classifiers.TensorflowClassifier(net, vocab=vocab, max_len=26, embedding=embedding_matrix)
print(classifier.get_pred(["i like apples", "i like apples"]))
print(classifier.get_prob(["i like apples", "i like apples"]))
print(classifier.get_grad(["i like apples", "i like apples"], [1, 1]))