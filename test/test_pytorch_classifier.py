import torch
import numpy as np
import pickle
import OpenAttack
import nltk
import torch.nn as nn
import unittest
import os

class SentimentRNN(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers, train_on_gpu, bidirectional=True, drop_prob=0.5):
        super(SentimentRNN, self).__init__()
        self.train_on_gpu = train_on_gpu
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=bidirectional)

        self.dropout = nn.Dropout(0.3)
        if(bidirectional):
            self.fc = nn.Linear(hidden_dim*2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        lstm_out, hidden = self.lstm(x, hidden)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size, -1, self.output_size)
        #print(sig_out.size())
        sig_out = torch.mean(sig_out, 1, True)
        #sig_out = sig_out[:, -1]
        #print(sig_out.squeeze())
        return sig_out.squeeze()
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        number = 1
        if(self.bidirectional):
            number = 2
        if(self.train_on_gpu):
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_())
        return hidden


class TestPytorch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OpenAttack.DataManager.set_path("./testdir")
        OpenAttack.DataManager.download("NLTKSentTokenizer")
        OpenAttack.DataManager.download("NLTKPerceptronPosTagger")
    
    @classmethod
    def tearDownClass(cls):
        os.system("rm -r ./testdir")
    
    def test_pytorch(self):
        punc = [',', '.', '?', '!']
        embedding_matrix = np.random.randn(20, 10).astype("float32")
        net = SentimentRNN(2, 10, 128, 2, False, False, 0.8).double()
        vocab = dict()
        num = 0
        for p in punc:
            vocab[p] = num
            num += 1
        for w in "i like apples".split():
            vocab[w] = num
            num += 1
        classifier = OpenAttack.classifiers.PytorchClassifier(net, word2id=vocab, max_len=250, embedding=embedding_matrix, token_pad=0)
        test_str = ["i like apples", "i like apples"]

        ret = classifier.get_pred(test_str)
        self.assertIsInstance(ret, np.ndarray)
        self.assertEqual(ret.shape, (len(test_str),))
        
        ret = classifier.get_prob(test_str)
        self.assertIsInstance(ret, np.ndarray)
        self.assertEqual(len(ret.shape), 2)
        self.assertEqual(ret.shape[0], len(test_str))
        
        ret = classifier.get_grad(test_str, [1, 1])
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
