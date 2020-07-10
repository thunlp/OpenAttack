from ..substitute import Substitute
from ..exceptions import NoEmbeddingException, WordNotInDictionaryException
import numpy as np

DEFAULT_CONFIG = {"cosine": False}


class EmbedBasedSubstitute(Substitute):
    def __init__(self, **kwargs):
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        if ("embedding" not in kwargs) or ("word2id" not in kwargs):
            raise NoEmbeddingException
        self.config["id2word"] = {
            val: key for key, val in self.config["word2id"].items()
        }
        if self.config["cosine"]:
            # normalize embedding if using cosine distance
            self.config["embedding"] = (
                self.config["embedding"]
                / np.linalg.norm(self.config["embedding"])[:, np.newaxis]
            )

    def __call__(self, word_or_char, pos=None, threshold=0.5):
        if word_or_char not in self.config["word2id"]:
            raise WordNotInDictionaryException
        wdid = self.config["word2id"][word_or_char]
        wdvec = self.config["embedding"][wdid]
        if self.config["cosine"]:
            dis = 1 - wdvec.dot(self.config["embedding"].T)
        else:
            dis = np.linalg.norm(self.config["embedding"] - wdvec, axis=1)
        rank = dis.argsort()
        ret = []
        for i in range(1, rank.shape[0]):
            if dis[rank[i]] > threshold:
                break
            ret.append((self.config["id2word"][rank[i]], dis[rank[i]]))
        return ret
