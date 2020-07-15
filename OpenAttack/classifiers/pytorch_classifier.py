import numpy as np
from . import ClassifierBase
from ..text_processors import DefaultTextProcessor
from ..utils import check_parameters
from ..exceptions import ClassifierNotSupportException

DEFAULT_CONFIG = {
    "device": None,
    "processor": DefaultTextProcessor(),
    "embedding": None,
    "word2id": None,
    "max_len": None,
    "tokenization": False,
    "padding": False,
    "token_unk": "<UNK>",
    "token_pad": "<PAD>",
    "require_length": False
}


class PytorchClassifier(ClassifierBase):
    def __init__(self, model, **kwargs):
        import torch
        self.model = model

        self.config = DEFAULT_CONFIG.copy()
        self.config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config.update(kwargs)
        self.to(self.config["device"])
        check_parameters(DEFAULT_CONFIG.keys(), self.config)

        super().__init__(**self.config)
        self.model.to(self.config["device"])
    
    def to(self, device):
        if isinstance(device, str):
            import torch
            device = torch.device(device)
        self.config["device"] = device
        self.model.to(device)
        return self

    def get_pred(self, input_):
        import torch
        input_, seq_len = self.preprocess(input_)
        if isinstance(input_, np.ndarray):
            input_ = torch.from_numpy(input_).to(self.config["device"])
        if self.config["tokenization"] and self.config["require_length"]:
            return self.model(input_, seq_len).max(dim=1)[1].cpu().numpy()
        else:
            return self.model(input_).max(dim=1)[1].cpu().numpy()
        

    def get_prob(self, input_):        
        import torch
        input_, seq_len = self.preprocess(input_)
        if isinstance(input_, np.ndarray):
            input_ = torch.from_numpy(input_).to(self.config["device"])
        
        if self.config["tokenization"] and self.config["require_length"]:
            return self.model(input_, seq_len).detach().cpu().numpy()
        else:
            return self.model(input_).detach().cpu().numpy()

    def get_grad(self, input_, labels):
        if self.config["word2id"] is None or self.config["embedding"] is None:
            raise ClassifierNotSupportException("gradient")

        import torch
        input_, seq_len = self.preprocess(input_)
        if isinstance(input_, np.ndarray):
            input_ = torch.from_numpy(input_).to(self.config["device"])
            input_.requires_grad_(True)
        if self.config["require_length"]:
            prob = self.model(input_, seq_len)
        else:
            prob = self.model(input_)
        loss = prob[ [ list(range(len(labels))), list(labels) ] ].sum()
        loss.backward()
        return prob.cpu().detach().numpy(), input_.grad.cpu().numpy()
