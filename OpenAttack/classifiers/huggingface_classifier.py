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


class HuggingfaceClassifier(ClassifierBase):
    def __init__(self, model, **kwargs):
        """
        :param transformers.Module model: Huggingface model for classification.
        :param str device: Device of pytorch model. **Default:** "cpu" if cuda is not available else "cuda"
        :param TextProcessor processor: Text processor used for tokenization. **Default:** :any:`DefaultTextProcessor`
        :param dict word2id: A dict maps token to index. If it's not None, torch.LongTensor will be passed to model. **Default:** None
        :param np.ndarray embedding: Word vector matrix of shape (vocab_size, vector_dim). If it's not None, torch.FloatTensor of shape (batch_size, max_input_len, vector_dim) will be passed to model.``word2id`` and ``embedding`` options are both required to support get_grad. **Default:** None
        :param int max_len: Max length of input tokens. If input token list is too long, it will be truncated. Uses None for no truncation. **Default:** None
        :param bool tokenization: If it's False, raw sentences will be passed to model, otherwise tokenized sentences will be passed. This option will be ignored if ``word2id`` is setted. **Default:** False
        :param bool padding: If it's True, add paddings to the end of sentences. This will be ignored if ``word2id`` option setted. **Default:** False
        :param str token_unk: Token for unknown tokens. **Default:** ``"<UNK>"``
        :param str token_unk: Token for padding. **Default:** ``"<PAD>"``
        :param bool require_length: If it's True, a list of lengths for each sentence will be passed to the model as the second parameter. **Default:** False

        :Package Requirements: * **pytorch**
        """
        import torch
        self.model = model

        self.config = DEFAULT_CONFIG.copy()
        self.config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config.update(kwargs)
        self.to(self.config["device"])
        self.hook = self.config["embedding"].register_forward_hook(self.__hook_fn)
        check_parameters(DEFAULT_CONFIG.keys(), self.config)

        super().__init__(**self.config)
        self.model.to(self.config["device"])

    def __hook_fn(self, module, input_, output_):
        self.curr_embedding = output_
        output_.retain_grad()
    
    def to(self, device):
        """
        :param str device: Device that moves model to.
        """
        if isinstance(device, str):
            import torch
            device = torch.device(device)
        self.config["device"] = device
        self.model.to(device)
        return self
        

    def get_prob(self, input_):        
        return self.model.predict(input_, [0] * len(input_))[0]

    def get_grad(self, input_, labels):
        return self.model.predict(input_, labels, tokenize=False)

    def predict(self,sen_list, labels=None):
        import torch
        
        sen_list = [
            sen[:self.config["max_len"] - 2] for sen in sen_list
        ]
        sent_lens = [ len(sen) for sen in sen_list ]
        attentions = np.array([
            [1] * (len(sen) + 2) + [0] * (self.config["max_len"] - 2 - len(sen))
            for sen in sen_list
        ], dtype='int64')
        sen_list = [
            [self.word2id[token] if token in self.word2id else self.word2id[self.config["token_unk"]] for token in sen]
             + [self.word2id[self.config["token_pad"]]] * (self.config["max_len"] - 2 - len(sen))
            for sen in sen_list
        ]
        tokeinzed_sen = np.array([
            [self.word2id["[CLS]"]] + sen + [self.word2id["[SEP]"]]
            for sen in sen_list
        ], dtype='int64')

        result = []
        result_grad = []
        
        if labels is None:
            labels = [0] * len(sen_list)
        labels = torch.LongTensor(labels).to(self.device)

        for i in range(len(tokeinzed_sen)):
            curr_sen = tokeinzed_sen[i]
            curr_mask = attentions[i]
            xs = torch.LongTensor([curr_sen]).to(self.device)
            masks = torch.LongTensor([curr_mask]).to(self.device)
            
            loss, logits = self.model(input_ids = xs,attention_mask = masks, labels=labels[i:i+1])
            logits = torch.nn.functional.softmax(logits,dim=-1)
            loss = - loss
            loss.backward()
            result_grad.append(self.curr_embedding.grad[0].clone())
            result.append(logits.cpu().detach().numpy()[0])
            self.curr_embedding.grad.zero_()

        max_len = max(sent_lens)
        result = np.array(result)
        result_grad = torch.stack(result_grad).cpu().numpy()[:, 1:1 + max_len]
        return result, result_grad
