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
    "token_pad": "<PAD>"
}


class PytorchClassifier(ClassifierBase):
    def __init__(self, model, **kwargs):
        import torch
        self.model = model

        self.config = DEFAULT_CONFIG.copy()
        self.config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config.update(kwargs)
        check_parameters(DEFAULT_CONFIG.keys(), self.config)

        super().__init__(**self.config)
        self.model.to(self.config["device"])

    def get_pred(self, input_):
        import torch
        input_ = self.preprocess(input_)
        if isinstance(input_, np.ndarray):
            input_ = torch.from_numpy(input_).to(self.config["device"])
        return self.model(input_).max(dim=1)[1].cpu().numpy()
        

    def get_prob(self, input_):        
        import torch
        input_ = self.preprocess(input_)
        if isinstance(input_, np.ndarray):
            input_ = torch.from_numpy(input_).to(self.config["device"])
        return self.model(input_).detach().cpu().numpy()

    def get_grad(self, input_, labels):
        if self.config["word2id"] is None or self.config["embedding"] is None:
            raise ClassifierNotSupportException("gradient")

        import torch
        input_ = self.preprocess(input_)
        if isinstance(input_, np.ndarray):
            input_ = torch.from_numpy(input_).to(self.config["device"])
            input_.requires_grad_(True)
        prob = self.model(input_)
        loss = prob[ [ list(range(len(labels))), list(labels) ] ].sum()
        loss.backward()
        return prob.cpu().detach().numpy(), input_.grad.cpu().numpy()
