from typing import Dict, Optional
from .base import WordSubstitute
from ....exceptions import WordNotInDictionaryException
import torch
from ....tags import *

DEFAULT_CONFIG = {"cosine": False}


class EmbedBasedSubstitute(WordSubstitute):
    """
    :param bool cosine: If true, uses cosine distance :math:`(1 - cos(v_a, v_b))`, otherwise uses Euclidian distance :math:`norm_2(v_a - v_b)`. **Default:** False.
    :param np.ndarray embedding: The 2d word vector matrix of shape (vocab_size, vector_dim).
    :param dict word2id: A dict maps word to index.

    A base class for all embedding-based substitute methods.
    
    An implementation of :py:class:`.WordSubstitute`.
    """

    TAGS = { * TAG_ALL_LANGUAGE }
    
    def __init__(self, word2id : Dict[str, int], embedding : torch.Tensor, cosine=False, k = 50, threshold = 0.5, device = None):
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
        """
        :param word: the raw word; 
        :param pos: part of speech of the word (`adj`, `adv`, `noun`, `verb`).
        :return: The result is a list of tuples, *(substitute, distance)*.
        :rtype: list of tuple

        In WordSubstitute, we return a list of words that are semantically similar to the original word.
        """
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
