from OpenAttack.classifier import Classifier
import numpy as np
from ..utils import check_parameters, HookCloser


DEFAULT_CONFIG = {
    "device": None,
    "embedding_layer": None,
    "token_pad": "[PAD]",
    "token_cls": "[CLS]",
    "token_sep": "[SEP]",
    "max_len": 128,
    "tokenizer": None,
    "batch_size": 8,
}


class HuggingfaceClassifier(Classifier):
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

        if self.config["embedding_layer"] != None:
            self.embedding = self.config["embedding_layer"].weight.detach().cpu().numpy()
        self.model.to(self.config["device"])

        self.__id_sep = self.config["tokenizer"].convert_tokens_to_ids(self.config["token_sep"])
        self.__id_cls = self.config["tokenizer"].convert_tokens_to_ids(self.config["token_cls"])
        self.__id_pad = self.config["tokenizer"].convert_tokens_to_ids(self.config["token_pad"])

    def to(self, device):
        """
        :param str device: Device that moves model to.
        """
        self.config["device"] = device
        self.model = self.model.to(device)
        return self
        
    def get_prob(self, input_):
        return self.get_grad([
            self.config["tokenizer"].tokenize(sent) for sent in input_
        ], [0] * len(input_))[0]

    def get_grad(self, input_, labels):
        v = self.predict(input_, labels)
        return v[0], v[1]

    def predict(self, sen_list, labels=None):
        import torch
        sen_list = [
            sen[:self.config["max_len"] - 2] for sen in sen_list
        ]
        sent_lens = [ len(sen) for sen in sen_list ]
        batch_len = max(sent_lens) + 2

        attentions = np.array([
            [1] * (len(sen) + 2) + [0] * (batch_len - 2 - len(sen))
            for sen in sen_list
        ], dtype='int64')
        sen_list = [
            self.config["tokenizer"].convert_tokens_to_ids(sen)
            for sen in sen_list
        ]
        tokeinzed_sen = np.array([
            [self.__id_cls] + sen + [self.__id_sep] + ([self.__id_pad] * (batch_len - 2 - len(sen)))
            for sen in sen_list
        ], dtype='int64')

        result = None
        result_grad = None
        all_hidden_states = None

        if labels is None:
            labels = [0] * len(sen_list)
        labels = torch.LongTensor(labels).to(self.config["device"])

        for i in range( (len(sen_list) + self.config["batch_size"] - 1) // self.config["batch_size"]):
            curr_sen = tokeinzed_sen[ i * self.config["batch_size"]: (i + 1) * self.config["batch_size"] ]
            curr_mask = attentions[ i * self.config["batch_size"]: (i + 1) * self.config["batch_size"] ]

            xs = torch.from_numpy(curr_sen).long().to(self.config["device"])
            masks = torch.from_numpy(curr_mask).long().to(self.config["device"])
            outputs = self.model(input_ids = xs,attention_mask = masks, output_hidden_states=True, labels=labels[ i * self.config["batch_size"]: (i + 1) * self.config["batch_size"] ])
            if i == 0:
                all_hidden_states = outputs.hidden_states[-1].detach().cpu()
                loss = outputs.loss
                logits = outputs.logits
                logits = torch.nn.functional.softmax(logits,dim=-1)
                loss = - loss
                loss.backward()
                if self.config["embedding_layer"] is not None:
                    result_grad = self.curr_embedding.grad.clone().cpu()
                    self.curr_embedding.grad.zero_()
                    self.curr_embedding = None
                result = logits.detach().cpu()
            else:
                all_hidden_states = torch.cat((all_hidden_states, outputs.hidden_states[-1].detach().cpu()), dim=0)
                loss = outputs.loss
                logits = outputs.logits
                logits = torch.nn.functional.softmax(logits,dim=-1)
                loss = - loss
                loss.backward()
                if self.config["embedding_layer"] is not None:
                    result_grad = torch.cat((result_grad, self.curr_embedding.grad.clone().cpu()), dim=0) 
                    self.curr_embedding.grad.zero_()
                    self.curr_embedding = None
                result = torch.cat((result, logits.detach().cpu()))

        result = result.numpy()
        all_hidden_states = all_hidden_states.numpy()
        if self.config["embedding_layer"] != None:
            result_grad = result_grad.numpy()[:, 1:-1]
        else:
            result_grad = None
        return result, result_grad, all_hidden_states

    def get_hidden_states(self, input_, labels=None):
        """
        :param list input_: A list of sentences of which we want to get the hidden states in the model.
        :rtype torch.tensor
        """
        return self.predict(input_, labels)[2]
