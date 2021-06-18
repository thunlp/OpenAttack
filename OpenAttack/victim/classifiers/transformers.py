import numpy as np
from .base import Classifier
from ...utils import language_by_name, HookCloser
from ...text_process.tokenizer import TransformersTokenizer
from ...attack_assist.word_embedding import WordEmbedding
import transformers
import torch

class TransformersClassifier(Classifier):

    @property
    def TAGS(self):
        if self.__lang_tag is None:
            return super().TAGS
        return super().TAGS.union({ self.__lang_tag })

    def __init__(self,
            model : transformers.PreTrainedModel,
            tokenizer : transformers.PreTrainedTokenizer,
            embedding_layer,
            device : torch.device = None, 
            max_length : int = 128,
            batch_size : int = 8,
            lang = None
        ):
        """
        Args:
            model: Huggingface model for classification.
            tokenizer: Huggingface tokenizer for classification. **Default:** None
            embedding_layer: The module of embedding_layer used in transformers models. For example, ``BertModel.bert.embeddings.word_embeddings``. **Default:** None
            device: Device of pytorch model. **Default:** "cpu" if cuda is not available else "cuda"
            max_len: Max length of input tokens. If input token list is too long, it will be truncated. Uses None for no truncation. **Default:** None
            batch_size: Max batch size of this classifier.
            lang: Language of this classifier. If is `None` then `TransformersClassifier` will intelligently select the language based on other parameters.

        """

        self.model = model

        if lang is not None:
            self.__lang_tag = language_by_name(lang)
        else:
            self.__lang_tag = None

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.to(device)

        self.curr_embedding = None
        self.hook = embedding_layer.register_forward_hook( HookCloser(self) )
        self.embedding_layer = embedding_layer

        self.word2id = dict()
        for i in range(tokenizer.vocab_size):
            self.word2id[tokenizer.convert_ids_to_tokens(i)] = i
        self.__tokenizer = tokenizer
        
        self.embedding = embedding_layer.weight.detach().cpu().numpy()

        self.token_unk = tokenizer.unk_token
        self.token_unk_id = tokenizer.unk_token_id

        self.max_length = max_length
        self.batch_size = batch_size
    
    @property
    def tokenizer(self):
        return TransformersTokenizer(self.__tokenizer, self.__lang_tag)

    def to(self, device : torch.device):
        """
        Args:
            device: Device that moves model to.
        """
        self.device = device
        self.model = self.model.to(device)
        return self
        
    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_):
        return self.get_grad([
            self.__tokenizer.tokenize(sent) for sent in input_
        ], [0] * len(input_))[0]

    def get_grad(self, input_, labels):
        v = self.predict(input_, labels)
        return v[0], v[1]

    def predict(self, sen_list, labels=None):
        sen_list = [
            sen[:self.max_length - 2] for sen in sen_list
        ]
        sent_lens = [ len(sen) for sen in sen_list ]
        batch_len = max(sent_lens) + 2

        attentions = np.array([
            [1] * (len(sen) + 2) + [0] * (batch_len - 2 - len(sen))
            for sen in sen_list
        ], dtype='int64')
        sen_list = [
            self.__tokenizer.convert_tokens_to_ids(sen)
            for sen in sen_list
        ]
        tokeinzed_sen = np.array([
            [self.__tokenizer.cls_token_id] + sen + [self.__tokenizer.sep_token_id] + ([self.__tokenizer.pad_token_id] * (batch_len - 2 - len(sen)))
            for sen in sen_list
        ], dtype='int64')

        result = None
        result_grad = None
        all_hidden_states = None

        if labels is None:
            labels = [0] * len(sen_list)
        labels = torch.LongTensor(labels).to(self.device)

        for i in range( (len(sen_list) + self.batch_size - 1) // self.batch_size):
            curr_sen = tokeinzed_sen[ i * self.batch_size: (i + 1) * self.batch_size ]
            curr_mask = attentions[ i * self.batch_size: (i + 1) * self.batch_size ]

            xs = torch.from_numpy(curr_sen).long().to(self.device)
            masks = torch.from_numpy(curr_mask).long().to(self.device)
            outputs = self.model(input_ids = xs,attention_mask = masks, output_hidden_states=True, labels=labels[ i * self.batch_size: (i + 1) * self.batch_size ])
            if i == 0:
                all_hidden_states = outputs.hidden_states[-1].detach().cpu()
                loss = outputs.loss
                logits = outputs.logits
                logits = torch.nn.functional.softmax(logits,dim=-1)
                loss = - loss
                loss.backward()
                
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
                
                result_grad = torch.cat((result_grad, self.curr_embedding.grad.clone().cpu()), dim=0) 
                self.curr_embedding.grad.zero_()
                self.curr_embedding = None

                result = torch.cat((result, logits.detach().cpu()))

        result = result.numpy()
        all_hidden_states = all_hidden_states.numpy()
        result_grad = result_grad.numpy()[:, 1:-1]
        return result, result_grad, all_hidden_states

    def get_hidden_states(self, input_, labels=None):
        """
        :param list input_: A list of sentences of which we want to get the hidden states in the model.
        :rtype torch.tensor
        """
        return self.predict(input_, labels)[2]
    
    def get_embedding(self):
        return WordEmbedding(self.word2id, self.embedding)
