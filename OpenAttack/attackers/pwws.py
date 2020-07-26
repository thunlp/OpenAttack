import numpy as np
from ..text_processors import DefaultTextProcessor
from ..substitutes import WordNetSubstitute
from ..utils import check_parameters
from ..attacker import Attacker
from ..exceptions import WordNotInDictionaryException

DEFAULT_CONFIG = {
    "threshold": 0.5,
    "processor": DefaultTextProcessor(),
    "substitute": None,
    "token_unk": "<UNK>"
}

class PWWSAttacker(Attacker):
    def __init__(self, **kwargs):
        """
        :param float threshold: Threshold used in substitute module. **Default:** 0.5
        :param WordSubstitute substitute: Substitute method used in this attacker. **Default:** :any:`WordNetSubstitute`
        :param TextProcessor processor: Text processor used in this attacker. **Default:** :any:`DefaultTextProcessor`
        :param str token_unk: A token which means "unknown token" in Classifier's vocabulary.

        :Classifier Capacity: Probability

        Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency. Shuhuai Ren, Yihe Deng, Kun He, Wanxiang Che. ACL 2019.
        `[pdf] <https://www.aclweb.org/anthology/P19-1103.pdf>`__
        `[code] <https://github.com/JHL-HUST/PWWS/>`__
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        if self.config["substitute"] is None:
            self.config["substitute"] = WordNetSubstitute()
        check_parameters(self.config.keys(), DEFAULT_CONFIG)

        self.processor = self.config["processor"]
        self.substitute = self.config["substitute"]
        
    def __call__(self, clsf, x_orig, target=None):
        x_orig = x_orig.lower()
        if target is None:
            targeted = False
            target = clsf.get_pred([x_orig])[0]  # calc x_orig's prediction
        else:
            targeted = True
        
        #x_orig = list(map(lambda x: x[0], self.processor.get_tokens(x_orig)))   # tokenize
        x_orig = self.processor.get_tokens(x_orig)
        poss =  list(map(lambda x: x[1], x_orig)) 
        x_orig =  list(map(lambda x: x[0], x_orig)) 

        S = self.get_saliency(clsf, x_orig, target, targeted) # (len(sent), )
        S_softmax = np.exp(S - S.max())
        S_softmax = S_softmax / S_softmax.sum()

        w_star = [ self.get_wstar(clsf, x_orig, i, poss[i], target, targeted) for i in range(len(x_orig)) ]  # (len(sent), )
        H = [ (idx, w_star[idx][0], S_softmax[idx] * w_star[idx][1]) for idx in range(len(x_orig)) ]

        H = sorted(H, key=lambda x:-x[2])

        ret_sent = x_orig.copy()
        for i in range(len(H)):
            idx, wd, _ = H[i]
            ret_sent[idx] = wd
            pred = clsf.get_pred([self.config["processor"].detokenizer(ret_sent)])[0]
            if targeted:
                if pred == target:
                    return (self.config["processor"].detokenizer(ret_sent), pred)
            else:
                if pred != target:
                    return (self.config["processor"].detokenizer(ret_sent), pred)
        return None


    
    def get_saliency(self, clsf, sent, target, targeted):
        x_hat_raw = []
        for i in range(len(sent)):
            left = sent[:i]
            right = sent[i + 1:]
            x_i_hat = left + [self.config["token_unk"]] + right
            x_hat_raw.append(self.config["processor"].detokenizer(x_i_hat))
        x_hat_raw.append(self.config["processor"].detokenizer(sent))
        res = clsf.get_prob(x_hat_raw)[:, target]
        if not targeted:
            res = res[-1] - res[:-1]
        else:
            res = res[:-1] - res[-1]
        return res

    def get_wstar(self, clsf, sent, idx, pos, target, targeted):
        word = sent[idx]
        try:
            rep_words = list(map(lambda x:x[0], self.substitute(word, pos, threshold = self.config["threshold"])))
        except WordNotInDictionaryException:
            rep_words = []
        rep_words = list(filter(lambda x: x != word, rep_words))
        if len(rep_words) == 0:
            return ( word, 0 )
        sents = []
        for rw in rep_words:
            new_sent = sent[:idx] + [rw] + sent[idx + 1:]
            sents.append(self.config["processor"].detokenizer(new_sent))
        sents.append(self.config["processor"].detokenizer(sent))
        res = clsf.get_prob(sents)[:, target]
        prob_orig = res[-1]
        res = res[:-1]
        if targeted:
            return (rep_words[ res.argmax() ],  res.max() - prob_orig )
        else:
            return (rep_words[ res.argmin() ],  prob_orig - res.min() )
    








