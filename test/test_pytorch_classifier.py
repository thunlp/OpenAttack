import torch
from torch.autograd import Variable
import numpy as np
import pickle
from TAADToolbox.classifiers.pytorch_classifier import PytorchClassifier
import nltk
import  torch.nn as nn

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, embedding_matrix, train_on_gpu, bidirectional=True, drop_prob=0.5):
        super(SentimentRNN, self).__init__()
        self.train_on_gpu = train_on_gpu
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=bidirectional)

        self.dropout = nn.Dropout(0.3)
        if(bidirectional):
            self.fc = nn.Linear(hidden_dim*2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        #x = x.long()
        #x = self.embedding(x)
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


embedding_matrix = np.load('embeddings_glove_50000.npy').transpose(1, 0)
net = SentimentRNN(50001, 2, 300, 128, 2, embedding_matrix,True, False, 0.8).double().cuda()
with open('imdb.vocab', 'r') as f:
    vocab_words = f.read().split('\n')
    vocab = dict([(w, i) for i, w in enumerate(vocab_words)])
lstm_classifier = PytorchClassifier(net, vocab=vocab, max_len=250, embedding=embedding_matrix)
print(lstm_classifier.get_pred(["i like apples", "i like apples"]))
print(lstm_classifier.get_prob(["i like apples", "i like apples"]))
print(lstm_classifier.get_grad(["i like apples", "i like apples"], [1, 1]))