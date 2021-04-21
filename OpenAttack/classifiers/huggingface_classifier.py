import numpy as np
from . import ClassifierBase
from ..text_processors import DefaultTextProcessor
from ..utils import check_parameters, HookCloser
from ..exceptions import ClassifierNotSupportException


DEFAULT_CONFIG = {
    "device": None,
    "embedding_layer": None,
    "token_pad": "[PAD]",
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
        :param str token_unk: Token for padding. **Default:** ``"<PAD>"``
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
        self.word2id = dict()
        if self.config["tokenizer"] != None:
            for i in range(self.config["tokenizer"].vocab_size):
                self.word2id[self.config["tokenizer"].convert_ids_to_tokens(i)] = i
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
        

    def get_grad(self, input_, labels):
        ret = self.predict(input_, labels)
        return ret[0], ret[1]

    def predict(self, sen_list, labels=None):
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
            [self.config["tokenizer"].convert_tokens_to_ids(token) for token in sen]
             + [self.config["tokenizer"].convert_tokens_to_ids(self.config["token_pad"])] * (self.config["max_len"] - 2 - len(sen))
            for sen in sen_list
        ]
        tokeinzed_sen = np.array([
            [self.config["tokenizer"].convert_tokens_to_ids("[CLS]")] + sen + [self.config["tokenizer"].convert_tokens_to_ids("[SEP]")]
            for sen in sen_list
        ], dtype='int64')

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

    def get_embedding(self):
        if self.config["embedding_layer"] != None:
            return self.config["embedding_layer"].weight
        else:
            return None