from ..text_processors import DefaultTextProcessor
from ..utils import check_parameters
from ..classifier import Classifier
from ..exceptions import WordNotInDictionaryException
import numpy as np

DEFAULT_CONFIG = {
    "processor": DefaultTextProcessor(),
    "embedding": None,
    "word2id": None,
    "max_len": None,
    "tokenization": False,
    "padding": False,
    "token_unk": "<UNK>",
    "token_pad": "<PAD>"
}

class ClassifierBase(Classifier):
    def __init__(self, **kwargs):
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        if self.config["word2id"] is not None:
            self.config["tokenization"] = True
            self.config["padding"] = True
        if self.config["embedding"] is not None:
            self.config["padding"] = True

    def pad_list(self, sent, pd, length):
        ret = sent.copy()
        while len(ret) < length:
            ret.append(pd)
        return ret
    
    def transform_id(self, word):
        if word in self.config["word2id"]:
            return self.config["word2id"][word]
        if isinstance(self.config["token_unk"], str) and self.config["token_unk"] in self.config["word2id"]:
            return self.config["word2id"][ self.config["token_unk"] ]
        if isinstance(self.config["token_unk"], int) and self.config["embedding"].shape[0] > self.config["token_unk"]:
            return self.config["token_unk"] 
        raise WordNotInDictionaryException(word)

    def preprocess(self, x_batch):
        if not self.config["tokenization"]:
            return x_batch, None
        
        x_batch = [ list(map(lambda x:x[0], self.config["processor"].get_tokens(sent))) for sent in x_batch ]
        seq_len = list( map( lambda x: len(x), x_batch ) )
        max_len = max( seq_len )
        if self.config["max_len"] is not None:
            max_len = min(max_len, self.config["max_len"])

        if self.config["word2id"] is None:
            if isinstance(self.config["token_pad"], str) and self.config["padding"]:
                x_batch = [ self.pad_list(tokens, self.config["token_pad"], max_len) for tokens in x_batch ]
            return x_batch, seq_len
        
        x_batch = [ list( map( lambda x: self.transform_id(x) , tokens) )  for tokens in x_batch ]
        
        if isinstance(self.config["token_pad"], str):
            pad_id = self.transform_id( self.config["token_pad"] )
        elif isinstance(self.config["token_pad"], int):
            pad_id = self.config["token_pad"]
        
        x_batch = [ self.pad_list(token_ids, pad_id, max_len) for token_ids in x_batch ]

        if self.config["embedding"] is None:
            return np.array(x_batch, dtype="int64"), seq_len
        
        x_batch = [ list( map( lambda idx: self.config["embedding"][idx] , token_ids) )  for token_ids in x_batch ]
        return np.array(x_batch, dtype="float64"), seq_len

    def preprocess_token(self, x_batch):
        
        seq_len = list( map( lambda x: len(x), x_batch ) )
        max_len = max( seq_len )
        if self.config["max_len"] is not None:
            max_len = min(max_len, self.config["max_len"])

        if self.config["word2id"] is None:
            if isinstance(self.config["token_pad"], str) and self.config["padding"]:
                x_batch = [ self.pad_list(tokens, self.config["token_pad"], max_len) for tokens in x_batch ]
            return x_batch, seq_len
        
        x_batch = [ list( map( lambda x: self.transform_id(x) , tokens) )  for tokens in x_batch ]
        
        if isinstance(self.config["token_pad"], str):
            pad_id = self.transform_id( self.config["token_pad"] )
        elif isinstance(self.config["token_pad"], int):
            pad_id = self.config["token_pad"]
        
        x_batch = [ self.pad_list(token_ids, pad_id, max_len) for token_ids in x_batch ]

        if self.config["embedding"] is None:
            return np.array(x_batch, dtype="int64"), seq_len
        
        x_batch = [ list( map( lambda idx: self.config["embedding"][idx] , token_ids) )  for token_ids in x_batch ]
        return np.array(x_batch, dtype="float64"), seq_len
