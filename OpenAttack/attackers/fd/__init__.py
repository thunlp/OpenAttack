from typing import Optional
import numpy as np
from ...text_process.tokenizer import Tokenizer, get_default_tokenizer
from ...utils import check_language, get_language, language_by_name
from ...attack_assist.substitute.word import WordSubstitute, get_default_substitute
from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...tags import TAG_English, Tag, TAG_ALL_LANGUAGE
from ...exceptions import WordNotInDictionaryException


class FDAttacker(ClassificationAttacker):
    @property
    def TAGS(self):
        return { self.__lang_tag, Tag("get_pred", "victim"), Tag("get_grad", "victim"), Tag("get_embedding", "victim") }

    def __init__(self,
            substitute : Optional[WordSubstitute] = None,
            tokenizer : Optional[Tokenizer] = None,
            threshold = 0.5,
            token_unk = "<UNK>",
            max_iter = 100,
            lang = None
        ):
        """
        :param TextProcessor processor: Text processor used in this attacker. **Default:** :any:`DefaultTextProcessor`
        :param WordSubstitute substitute: Substitute method used in this attacker. **Default:** :any:`CounterFittedSubstitute`
        :param dict word2id: A dict that maps tokens to ids.
        :param np.ndarray embedding: The 2d word vector matrix of shape (vocab_size, vector_dim).
        :param token_unk: The word_id or the token for out-of-vocabulary words. **Default:** ``"<UNK>"``.
        :type token_unk: int or str
        :param float threshold: Threshold for substitute module. **Default:** 0.5.
        :param int max_iter: Maximum number of iterations in FDAttacker.

        :Classifier Capacity: Gradient

        Crafting Adversarial Input Sequences For Recurrent Neural Networks. Nicolas Papernot, Patrick McDaniel, Ananthram Swami, Richard Harang. MILCOM 2016.
        `[pdf] <https://arxiv.org/pdf/1604.08275.pdf>`__
        """

        if substitute is not None and tokenizer is not None:
            self.__lang_tag = get_language([substitute, tokenizer])
        if substitute is not None:
            self.__lang_tag = get_language([substitute])
        elif tokenizer is not None:
            self.__lang_tag = get_language([tokenizer])
        else:
            if lang is None:
                self.__lang_tag = TAG_English
            else:
                self.__lang_tag = language_by_name(lang)
                if self.__lang_tag is None:
                    raise ValueError("Unknown language `%s`" % lang)

        if substitute is None:
            substitute = get_default_substitute(self.__lang_tag)
        
        self.substitute = substitute

        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer

        check_language([self.tokenizer, self.substitute], self.__lang_tag)

        self.threshold = threshold
        self.token_unk = token_unk
        self.max_iter = max_iter
    
    def attack(self, victim: Classifier, x_orig, goal: ClassifierGoal):
        x_orig = x_orig.lower()
        
        sent = self.tokenizer.tokenize(x_orig, pos_tagging=False)

        victim_embedding = victim.get_embedding()
        
        for i in range(self.max_iter):
            curr_sent = self.tokenizer.detokenize(sent)
            pred = victim.get_pred([ curr_sent ])[0]
            if goal.check(curr_sent, pred):
                return curr_sent
            
            iter_cnt = 0
            while True:
                idx = np.random.choice(len(sent))
                iter_cnt += 1
                if iter_cnt > 5 * len(sent):    # Failed to find a substitute word
                    return None
                try:
                    reps = list(map(lambda x:x[0], self.substitute(sent[idx], None)))
                except WordNotInDictionaryException:
                    continue
                reps = list(filter(lambda x: x in victim_embedding.word2id, reps))
                if len(reps) > 0:
                    break
            
            prob, grad = victim.get_grad([sent], [goal.target])
            grad = grad[0]
            prob = prob[0]
            if grad.shape[0] != len(sent) or grad.shape[1] != victim_embedding.embedding.shape[1]:
                raise RuntimeError("Sent %d != Gradient %d" % (len(sent), grad.shape[0]))
            s1 = np.sign(grad[idx])
            
            mn = None
            mnwd = None
            
            for word in reps:
                s0 = np.sign(victim_embedding.transform(word, self.token_unk) - victim_embedding.transform(sent[idx], self.token_unk))
                v = np.abs(s0 - s1).sum()
                if goal.targeted:
                    v = -v
                
                if (mn is None) or v < mn:
                    mn = v
                    mnwd = word

            if mnwd is None:
                return None
            sent[idx] = mnwd
        return None
