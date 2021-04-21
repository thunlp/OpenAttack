import numpy as np
from . import ClassifierBase
from ..text_processors import DefaultTextProcessor
from ..utils import check_parameters, HookCloser
from ..exceptions import ClassifierNotSupportException


DEFAULT_CONFIG = {
    "device": None,
    "embedding_layer": None,
    "max_len": None,
    "tokenizer": None,
}


class HuggingfaceClassifier(ClassifierBase):
    def __init__(self, model, **kwargs):
        """
        :param transformers.Module model: Huggingface model for classification.
        :param str device: Device of pytorch model. **Default:** "cpu" if cuda is not available else "cuda"
        :param transformers.model.embeddings.word_embeddings embedding_layer: The module of embedding_layer used in transformers models. For example, ``BertModel.bert.embeddings.word_embeddings``. ``word2id`` and ``embedding`` options are both required to support get_grad. **Default:** None
        :param transformers.Tokenizer tokenizer: Huggingface tokenizer for classification. **Default:** None
        :param int max_len: Max length of input tokens. If input token list is too long, it will be truncated. Uses None for no truncation. **Default:** None

        :Package Requirements: * **pytorch**
        """
        import torch
        self.model = model

        self.config = DEFAULT_CONFIG.copy()
        self.config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config.update(kwargs)
        self.to(self.config["device"])
        if self.config["embedding_layer"] != None:
            self.curr_embedding = None
            self.hook = self.config["embedding_layer"].register_forward_hook( HookCloser(self) )
        check_parameters(DEFAULT_CONFIG.keys(), self.config)

        super().__init__(**self.config)
        self.model.to(self.config["device"])

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
        return self.predict(input_, [0] * len(input_))[0]

    def get_grad(self, input_, labels):
        ret = self.predict(input_, labels)
        return ret[0], ret[1]

    def predict(self, sen_list, labels=None):
        import torch
        sent_lens = [ len(sen) for sen in sen_list ]
        tokenized_batch = self.config["tokenizer"](sen_list, padding=True, truncation=True, max_length=self.config["max_len"], return_tensors="pt")
        tokeinzed_sen = tokenized_batch["input_ids"].numpy().tolist()
        attentions = tokenized_batch["attention_mask"].numpy().tolist()
        result = []
        result_grad = []
        
        if labels is None:
            labels = [0] * len(sen_list)
        labels = torch.LongTensor(labels).to(self.config["device"])

        for i in range(len(tokeinzed_sen)):
            curr_sen = tokeinzed_sen[i]
            curr_mask = attentions[i]
            xs = torch.LongTensor([curr_sen]).to(self.config["device"])
            masks = torch.LongTensor([curr_mask]).to(self.config["device"])
            
            outputs = self.model(input_ids = xs,attention_mask = masks, output_hidden_states=True, labels=labels[i:i+1])
            all_hidden_states = outputs.hidden_states
            loss = outputs.loss
            logits = outputs.logits
            logits = torch.nn.functional.softmax(logits,dim=-1)
            loss = - loss
            loss.backward()
            if self.config["embedding_layer"] != None:
                result_grad.append(self.curr_embedding.grad[0].clone())
                self.curr_embedding.grad.zero_()
            result.append(logits.cpu().detach().numpy()[0])
            

        max_len = max(sent_lens)
        result = np.array(result)
        if self.config["embedding_layer"] != None:
            result_grad = torch.stack(result_grad).cpu().numpy()[:, 1:1 + max_len]
        else:
            result_grad = 0
        return result, result_grad, all_hidden_states

    def get_hidden_states(self, input_, labels=None):
        return self.predict(input_, labels)[2]
