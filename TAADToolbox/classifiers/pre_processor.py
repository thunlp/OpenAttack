from ..text_processors import DefaultTextProcessor
from ..utils import check_parameters
import numpy as np


DEFAULT_CONFIG = {
    "processor": DefaultTextProcessor(),
    "embedding": None,
}

class PreProcessor(object):
    def __init__(self, *args, **kwargs):
        self.vocab = args[0]
        self.max_len = args[1]
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        if np.any(self.config["embedding"]) != None:
            self.use_embedding = True
        check_parameters(DEFAULT_CONFIG.keys(), self.config)

    def POS_process(self, input_):
        seqs = []
        for sentence in input_:
            ret = self.config["processor"].get_tokens(sentence.lower())
            seq = []
            for (key, value) in ret:
                if len(seq) < self.max_len:
                    seq.append(self.vocab[key])
            while len(seq) < self.max_len:
                seq.append(0)
            seqs.append(seq)
        return seqs

    def embedding_process(self, input_):
        embedding_dim = self.config["embedding"].shape[1]
        embeds = np.zeros(shape=((len(input_), self.max_len, embedding_dim)))
        for i in range(len(input_)):
            for j in range(self.max_len):
                embeds[i, j, :] = self.config["embedding"][input_[i][j]]
        return embeds

