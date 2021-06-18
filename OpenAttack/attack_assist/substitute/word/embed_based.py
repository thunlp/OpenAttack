from typing import Dict, Optional
from .base import WordSubstitute
from ....exceptions import WordNotInDictionaryException
import torch
from ....tags import *

DEFAULT_CONFIG = {"cosine": False}


class EmbedBasedSubstitute(WordSubstitute):
    
    def __init__(self, word2id : Dict[str, int], embedding : torch.Tensor, cosine=False, k = 50, threshold = 0.5, device = None):
        """
        Embedding based word substitute.

        Args:
            word2id: A `dict` maps words to indexes.
            embedding: A word embedding matrix.
            cosine: If `true` then the cosine distance is used, otherwise the Euclidian distance is used.
            threshold: Distance threshold. Default: 0.5
            k: Top-k results to return. If k is `None`, all results will be returned. Default: 50
            device: A pytocrh device for computing distances. Default: "cpu"
        
        """

        if device is None:
            device = "cpu"
            
        self.word2id = word2id
        self.embedding = embedding
        self.cosine = cosine
        self.k = k
        self.threshold = threshold

        self.id2word = {
            val: key for key, val in self.word2id.items()
        }
        
        if cosine:
            self.embedding = self.embedding / self.embedding.norm(dim=1, keepdim=True)
        
        self.embedding = self.embedding.to(device)
    
    def __call__(self, word: str, pos: Optional[str] = None):
        return self.substitute(word, pos)
    
    def substitute(self, word, pos):
        if word not in self.word2id:
            raise WordNotInDictionaryException()
        wdid = self.word2id[word]
        wdvec = self.embedding[wdid, :]
        if self.cosine:
            dis = 1 - (wdvec * self.embedding).sum(dim=1)
        else:
            dis = (wdvec - self.embedding).norm(dim=1)

        idx = dis.argsort()
        if self.k is not None:
            idx = idx[:self.k]
        
        threshold_end = 0
        while threshold_end < len(idx) and dis[idx[threshold_end]] < self.threshold:
            threshold_end += 1
        idx = idx[:threshold_end].tolist()
        return [
            (self.id2word[id_], dis[id_].item()) for id_ in idx
        ]
