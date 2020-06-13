import numpy as np
import torch
import torch.nn as nn
from .pre_processor import PreProcessor
from torch.autograd import Variable
from ..classifier import Classifier
from ..text_processors import DefaultTextProcessor
from ..utils import check_parameters
from ..exceptions import ClassifierNotSupportException

DEFAULT_CONFIG = {
    "device": 'cpu',
    "processor": DefaultTextProcessor(),
    "POS": False,
    "embedding": None,
    "use_sentence": False,
    "use_word_id": False,
    "use_embedding": True,
}


class PytorchClassifier(Classifier):
    def __init__(self, *args, **kwargs):
        self.model = args[0]
        self.dict = args[1]
        self.max_len = args[2]
        
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        check_parameters(DEFAULT_CONFIG.keys(), self.config)
        if self.config["use_sentence"]:
            self.use_sentence = True
        else:
            self.use_sentence = False
        if self.config["use_word_id"]:
            self.use_word_id = True
        else:
            self.use_word_id = False
        if self.config["use_embedding"]:
            self.use_embedding = True
        else:
            self.use_embedding = False
        if kwargs["device"] != "cpu":
            self.use_gpu = True
        else:
            self.use_gpu = False
        self.pre_processor = PreProcessor(args[1], args[2], processor=self.config["processor"], embedding=self.config["embedding"])

    def get_pred(self, input_):
        if self.use_sentence:
            prob = self.model(input_)
        elif self.use_word_id:
            seqs = torch.tensor(self.pre_processor.POS_process(input_))
            if self.use_gpu:
                seqs =seqs.cuda()
            prob = self.model(seqs)
        elif self.use_embedding:
            seqs = self.pre_processor.POS_process(input_)
            seqs2 = torch.from_numpy(self.pre_processor.embedding_process(seqs))
            if self.use_gpu:
                seqs2 =seqs2.cuda()
            prob = self.model(seqs2)
        return np.array(prob.max(1)[1].cpu(), dtype=np.long)
        

    def get_prob(self, input_):
        if self.use_sentence:
            prob = self.model(input_)
        elif self.use_word_id:
            seqs = torch.tensor(self.pre_processor.POS_process(input_))
            if self.use_gpu:
                seqs =seqs.cuda()
            prob = self.model(seqs)
        elif self.use_embedding:
            seqs = self.pre_processor.POS_process(input_)
            seqs2 = torch.from_numpy(self.pre_processor.embedding_process(seqs))
            if self.use_gpu:
                seqs2 =seqs2.cuda()
            prob = self.model(seqs2)
        return prob.cpu().detach().numpy()

    def get_grad(self, input_, labels):
        if self.use_sentence:
            raise ClassifierNotSupportException
        elif self.use_word_id:
            raise ClassifierNotSupportException
        elif self.use_embedding:
            seqs = self.pre_processor.POS_process(input_)
            seqs2 = torch.from_numpy(self.pre_processor.embedding_process(seqs))
            if self.use_gpu:
                seqs2 =seqs2.cuda()
            seqs2 = Variable(seqs2, requires_grad=True)
            prob = self.model(seqs2)
        if self.config["device"] == "gpu":
            loss = torch.zeros([1]).cuda()
        else:
            loss = torch.zeros([1])
        for i in range(len(labels)):
            loss += prob[i][labels[i]]
        if self.config["device"] == "gpu":
            sample = torch.zeros(loss.size()).cuda()
        else:
            sample = torch.zeros(loss.size())
        loss.backward(sample)
        return seqs2.grad.cpu().numpy()