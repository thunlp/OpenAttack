"""
:type: PytorchClassifier
:Size: 1.683MB
:Data Requirements: :py:data:`.AttackAssist.GloVe`
:Package Requirements:
    * pytorch

Pretrained BiLSTM model on SST-2 dataset. See :py:data:`Dataset.SST` for detail.
"""

NAME = "Victim.BiLSTM.SST"
DOWNLOAD = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/victim/bilstm_sst.pth"

def LOAD(path):
    import torch
    import torch.nn as nn
    import OpenAttack

    class Model(nn.Module):
        def __init__(self, input_dim=300, output_dim=2, hidden_size=128, bidirectional=True):
            super(Model, self).__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_size = hidden_size * (2 if bidirectional else 1)
            self.lstm = nn.LSTM(self.input_dim, hidden_size, batch_first=True, bidirectional=bidirectional)
            self.clsf = nn.Linear(self.hidden_size, self.output_dim)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x, seq_len):
            """
            :x:         (batch, max_len, input_dim)
            """
            seq = x.float()
            batch_size = seq.size(0)
            max_len = seq.size(1)
            device = x.device

            lstm_out, _ = self.lstm(seq)   # (batch, seq_len, hidden_size)

            mask = torch.arange(max_len, device=device).repeat(batch_size, 1).lt(torch.LongTensor(seq_len).to(device).unsqueeze(dim=-1).repeat(1, max_len)).float().unsqueeze(dim=-1).repeat(1, 1, self.hidden_size)
            # mean pooling
            lstm_out = lstm_out * mask + ( -1e6 * (1 - mask) )
            lstm_out = (lstm_out).max(dim=1)[0]    # (batch, hidden_size)
            
            out = self.clsf(lstm_out)   # (batch, 2)
            return self.softmax(out)
    model = Model()
    model.load_state_dict( torch.load(path, map_location=lambda storage, loc: storage) )
    
    word_vector = OpenAttack.DataManager.load("AttackAssist.GloVe")

    return OpenAttack.classifiers.PytorchClassifier(model, 
        word2id=word_vector.word2id, embedding=word_vector.get_vecmatrix(), 
        token_unk= "UNK", require_length=True, device="cpu")
