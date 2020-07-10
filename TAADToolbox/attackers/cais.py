import numpy as np
from ..text_processors import DefaultTextProcessor
from ..substitutes import CounterFittedSubstitute
from ..utils import check_parameters, detokenizer
from ..exceptions import WordEmbeddingRequired, WordNotInDictionaryException, TokensNotAligned
from ..attacker import Attacker

DEFAULT_CONFIG = {
    "processor": DefaultTextProcessor(),
    "substitute": None,
    "embedding": None,
    "wordid": None,
    "threshold": 0.5,
    "token_unk": "<UNK>",
    "max_iter": 10
}


class CAISAttacker(Attacker):
    def __init__(self, **kwargs):
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        if self.config["substitute"] is None:
            self.config["substitute"] = CounterFittedSubstitute()
        if ((self.config["embedding"] is None) or (self.config["wordid"] is None)):
            raise WordEmbeddingRequired()
        check_parameters(DEFAULT_CONFIG.keys(), self.config)
        self.processor = self.config["processor"]
        self.embedding = self.config["embedding"]
        self.wordid = self.config["wordid"]
        self.substitute = self.config["substitute"]
    
    def __call__(self, clsf, x_orig, target=None):
        x_orig = x_orig.lower()
        if target is None:
            targeted = False
            target = clsf.get_pred([x_orig])[0]
        else:
            targeted = True
        sent = list(map(lambda x: x[0], self.processor.get_tokens(x_orig)))
        
        for i in range(self.config["max_iter"]):
            pred = clsf.get_pred([ detokenizer(sent) ])[0]
            if targeted:
                if pred == target:
                    return (detokenizer(sent), pred)
            else:
                if pred != target:
                    return (detokenizer(sent), pred)
            
            iter_cnt = 0
            while True:
                idx = np.random.choice(len(sent))
                iter_cnt += 1
                if iter_cnt > 5 * len(sent):    # Failed to find a substitute word
                    return None
                try:
                    reps = list(map(lambda x:x[0], self.substitute(sent[idx], threshold=self.config["threshold"])))
                except WordNotInDictionaryException:
                    continue
                reps = list(filter(lambda x: x in self.wordid, reps))
                if len(reps) > 0:
                    break
            
            prob, grad = clsf.get_grad([detokenizer(sent)], [target])
            grad = grad[0]
            prob = prob[0]
            if grad.shape[0] != len(sent) or grad.shape[1] != self.embedding.shape[1]:
                raise TokensNotAligned("Sent %d != Gradient %d" % (len(sent), grad.shape[0]))
            s1 = np.sign(grad[idx])
            
            mn = None
            mnwd = None
            
            for word in reps:
                s0 = np.sign(self.transform(word) - self.transform(sent[idx]))
                v = np.abs(s0 - s1).sum()
                if targeted:
                    v = -v
                
                if (mn is None) or v < mn:
                    mn = v
                    mnwd = word

            if mnwd is None:
                return None
            sent[idx] = mnwd
        return None

    def transform(self, word):
        if word in self.wordid:
            return self.embedding[ self.wordid[word] ]
        else:
            if isinstance(self.config["token_unk"], int):
                return self.embedding[ self.config["token_unk"] ]
            else:
                return self.embedding[ self.wordid[ self.config["token_unk"] ] ]
        
