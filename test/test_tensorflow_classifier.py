import tensorflow as tf 
import numpy as np
import pickle
import OpenAttack
import nltk
import unittest
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class TestTensorflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OpenAttack.DataManager.set_path("./testdir")
        OpenAttack.DataManager.download("TProcess.NLTKSentTokenizer")
        OpenAttack.DataManager.download("TProcess.NLTKPerceptronPosTagger")
        cls.dp = OpenAttack.text_processors.DefaultTextProcessor()
    
    @classmethod
    def tearDownClass(cls):
        os.system("rm -r ./testdir")
    
    def test_tensorflow(self):
        tf.keras.backend.set_floatx('float64')
        net = tf.keras.models.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300)),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(2, activation='sigmoid')
        ])

        punc = [',', '.', '?', '!']
        embedding_matrix = np.random.randn(20, 10).astype("float32")
        vocab = dict()
        num = 0
        for p in punc:
            vocab[p] = num
            num += 1
        for w in "i like apples".split():
            vocab[w] = num
            num += 1
        classifier =  OpenAttack.classifiers.TensorflowClassifier(net, word2id=vocab, max_len=26, embedding=embedding_matrix, token_pad=0)
        processor = OpenAttack.text_processors.DefaultTextProcessor()
        test_str = ["i like apples", "i like apples"]

        ret = classifier.get_pred(test_str)
        self.assertIsInstance(ret, np.ndarray)
        self.assertEqual(ret.shape, (len(test_str),))
        
        ret = classifier.get_prob(test_str)
        self.assertIsInstance(ret, np.ndarray)
        self.assertEqual(len(ret.shape), 2)
        self.assertEqual(ret.shape[0], len(test_str))
        
        x_batch = [ list(map(lambda x:x[0], processor.get_tokens(sent))) for sent in test_str ]
        ret = classifier.get_grad(x_batch, [1, 1])
        self.assertIsInstance(ret, tuple)
        self.assertEqual(len(ret), 2)
        self.assertIsInstance(ret[0], np.ndarray)
        self.assertEqual(len(ret[0].shape), 2)
        self.assertEqual(ret[0].shape[0], len(test_str))
        self.assertIsInstance(ret[1], np.ndarray)
        self.assertEqual(len(ret[1].shape), 3)
        self.assertEqual(ret[1].shape[0], len(test_str))
        self.assertEqual(ret[1].shape[1], len(test_str[0].split()))
        self.assertEqual(ret[1].shape[2], embedding_matrix.shape[1])