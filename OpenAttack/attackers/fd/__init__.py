from typing import List, Optional
import numpy as np
from ...text_process.tokenizer import Tokenizer, get_default_tokenizer
from ...utils import check_language, get_language, language_by_name
from ...attack_assist.substitute.word import WordSubstitute, get_default_substitute
from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...tags import TAG_English, Tag
from ...exceptions import WordNotInDictionaryException
from ...attack_assist.filter_words import get_default_filter_words

class FDAttacker(ClassificationAttacker):
    @property
    def TAGS(self):
        return { self.__lang_tag, Tag("get_pred", "victim"), Tag("get_grad", "victim"), Tag("get_embedding", "victim") }

    def __init__(self,
            substitute : Optional[WordSubstitute] = None,
            tokenizer : Optional[Tokenizer] = None,
            token_unk : str = "<UNK>",
            max_iter : int = 100,
            lang : Optional[str] = None,
            filter_words : List[str] = None
        ):
        """
        Crafting Adversarial Input Sequences For Recurrent Neural Networks. Nicolas Papernot, Patrick McDaniel, Ananthram Swami, Richard Harang. MILCOM 2016.
        `[pdf] <https://arxiv.org/pdf/1604.08275.pdf>`__

        Args:
            substitute: A substitute that will be used during the attack procedure. Must be an instance of :py:class:`.WordSubstitute`
            tokenizer: A tokenizer that will be used during the attack procedure. Must be an instance of :py:class:`.Tokenizer`
            token_unk: The token id or the token name for out-of-vocabulary words in victim model. **Default:** ``"<UNK>"``
            max_iter: Maximum number of iterations in attack procedure.
            lang: The language used in attacker. If is `None` then `attacker` will intelligently select the language based on other parameters.
            filter_words: A list of words that will be preserved in the attack procesudre.

        :Classifier Capacity:
            * get_pred
            * get_grad
            * get_embedding
        
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

        if filter_words is None:
            filter_words = get_default_filter_words(self.__lang_tag)
        self.filter_words = set(filter_words)

        check_language([self.tokenizer, self.substitute], self.__lang_tag)

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
                if sent[idx] in self.filter_words:
                    continue
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
