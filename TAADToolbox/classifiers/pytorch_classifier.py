import numpy as np
from .pre_processor import PreProcessor
from ..classifier import Classifier
from ..text_processors import DefaultTextProcessor
from ..utils import check_parameters
from ..exceptions import ClassifierNotSupportException

DEFAULT_CONFIG = {
    "device": None,
    "processor": DefaultTextProcessor(),
    "embedding": None,
    "vocab": None,
    "max_len":None,    
    "use_sentence": False,
    "use_word_id": False,
    "use_embedding": True,
}


class PytorchClassifier(Classifier):
    def __init__(self, *args, **kwargs):
        import torch

        self.model = args[0]
        
        self.config = DEFAULT_CONFIG.copy()
        self.config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config.update(kwargs)
        check_parameters(DEFAULT_CONFIG.keys(), self.config)
        self.use_sentence = self.config["use_sentence"]
        self.use_word_id = self.config["use_word_id"]
        self.use_embedding = self.config["use_embedding"]
        if self.use_word_id or self.use_embedding:
            self.pre_processor = PreProcessor(self.config["vocab"], self.config["max_len"], processor=self.config["processor"], embedding=self.config["embedding"])
        self.model.to(self.config["device"])

    def get_pred(self, input_):
        import torch

        if self.use_sentence:
            prob = self.model(input_)
        elif self.use_word_id:
            seqs = torch.tensor(self.pre_processor.POS_process(input_), device=self.config["device"])
            prob = self.model(seqs)
        elif self.use_embedding:
            seqs = self.pre_processor.POS_process(input_)
            seqs2 = torch.tensor(self.pre_processor.embedding_process(seqs), device=self.config["device"])
            prob = self.model(seqs2)
        return np.array(prob.max(1)[1].cpu(), dtype=np.long)
        

    def get_prob(self, input_):
        import torch

        if self.use_sentence:
            prob = self.model(input_)
        elif self.use_word_id:
            seqs = torch.tensor(self.pre_processor.POS_process(input_), device=self.config["device"])
            prob = self.model(seqs)
        elif self.use_embedding:
            seqs = self.pre_processor.POS_process(input_)
            seqs2 = torch.tensor(self.pre_processor.embedding_process(seqs), device=self.config["device"])
            prob = self.model(seqs2)
        return prob.cpu().detach().numpy()

    def get_grad(self, input_, labels):
        import torch
        
        if self.use_sentence:
            raise ClassifierNotSupportException
        elif self.use_word_id:
            raise ClassifierNotSupportException
        elif self.use_embedding:
            seqs = self.pre_processor.POS_process(input_)
            seqs2 = torch.tensor(self.pre_processor.embedding_process(seqs), device=self.config["device"], requires_grad=True)
            prob = self.model(seqs2)
        loss = torch.zeros([1], device=self.config["device"])
        for i in range(len(labels)):
            loss += prob[i][labels[i]]
        sample = torch.zeros(loss.size(), device=self.config["device"])
        loss.backward(sample)
        return seqs2.grad.cpu().numpy()