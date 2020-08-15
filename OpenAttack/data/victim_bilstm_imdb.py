"""
:type: PytorchClassifier
:Size: 21.8MB
:Package Requirements:
    * pytorch

Pretrained BiLSTM model on IMDB dataset. See :py:data:`Dataset.IMDB` for detail.
"""

NAME = "Victim.BiLSTM.IMDB"
DOWNLOAD = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/victim/bilstm_imdb.pth"


def LOAD(path):
    import torch
    import torch.nn as nn
    import OpenAttack

    class SentimentRNN(nn.Module):
        def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, bidirectional=True,
                     drop_prob=0.5):
            super(SentimentRNN, self).__init__()
            self.output_size = output_size
            self.n_layers = n_layers
            self.hidden_dim = hidden_dim
            self.bidirectional = bidirectional
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                                dropout=drop_prob, batch_first=True,
                                bidirectional=bidirectional)
            self.dropout = nn.Dropout(0.3)
            if bidirectional:
                self.fc = nn.Linear(hidden_dim * 2, output_size)
            else:
                self.fc = nn.Linear(hidden_dim, output_size)
            self.sig = nn.Sigmoid()

        def forward(self, x, hidden):
            batch_size = x.size(0)
            x = x.long()
            embeds = self.embedding(x)
            lstm_out, hidden = self.lstm(embeds, hidden)
            out = self.dropout(lstm_out)
            out = self.fc(out)
            sig_out = self.sig(out)
            sig_out = sig_out.view(batch_size, -1)
            sig_out = sig_out[:, -1]
            return sig_out, hidden

        def init_hidden(self, batch_size):
            device = 'cpu'
            train_on_gpu = torch.cuda.is_available()
            weight = next(self.parameters()).data
            number = 1
            if self.bidirectional:
                number = 2
            if train_on_gpu:
                w1 = weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().to(device)
                w2 = weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().to(device)
                hidden = (w1, w2)
            else:
                hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_(),
                          weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_())
            return hidden

    model = SentimentRNN(9998, 1, 300, 256, 2, True)
    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    return OpenAttack.classifiers.PytorchClassifier(model, token_unk="UNK", require_length=True, device="cpu")
