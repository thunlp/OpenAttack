import numpy as np
from ..text_processors import DefaultTextProcessor
from ..substitutes import CounterFittedSubstitute
from ..exceptions import WordNotInDictionaryException
from ..utils import check_parameters
from ..attacker import Attacker


DEFAULT_SKIP_WORDS = set(
    [
        "the",
        "and",
        "a",
        "of",
        "to",
        "is",
        "it",
        "in",
        "i",
        "this",
        "that",
        "was",
        "as",
        "for",
        "with",
        "movie",
        "but",
        "film",
        "on",
        "not",
        "you",
        "he",
        "are",
        "his",
        "have",
        "be",
    ]
)

DEFAULT_CONFIG = {
    "skip_words": DEFAULT_SKIP_WORDS,
    "neighbour_threshold": 0.8,
    "top_n": 20,
    "processor": DefaultTextProcessor(),
    "substitute": None,
}

class HotFlipAttacker(Attacker):
    def __init__(self, **kwargs):
        """
        :param list skip_words: A list of words which won't be replaced during the attack. **Default:** A list of words that is most frequently used.
        :param float neighbour_threshold: Threshold used in substitute module. **Default:** 0.8
        :param int top_n: Maximum candidates of word substitution. **Default:** 20
        :param TextProcessor processor: Text processor used in this attacker. **Default:** :any:`DefaultTextProcessor`
        :param WordSubstitute substitute: Substitute method used in this attacker. **Default:** :any:`CounterFittedSubstitute`

        :Classifier Capacity: Gradient

        HotFlip: White-Box Adversarial Examples for Text Classification. Javid Ebrahimi, Anyi Rao, Daniel Lowd, Dejing Dou. ACL 2018.
        `[pdf] <https://www.aclweb.org/anthology/P18-2006>`__
        `[code] <https://github.com/AnyiRao/WordAdver>`__
        
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        if self.config["substitute"] is None:
            self.config["substitute"] = CounterFittedSubstitute()

        check_parameters(DEFAULT_CONFIG.keys(), self.config)

    def __call__(self, clsf, x_orig, target=None):
        """
        * **clsf** : **Classifier** .
        * **x_orig** : Input sentence.
        """
        x_orig = x_orig.lower()
        if target is None:
            targeted = False
            target = clsf.get_pred([x_orig])[0]  
        else:
            targeted = True

        x_orig = self.config["processor"].get_tokens(x_orig)
        x_pos =  list(map(lambda x: x[1], x_orig))
        x_orig = list(map(lambda x: x[0], x_orig))
        counter = -1
        for word, pos in zip(x_orig, x_pos):
            counter += 1
            if word in self.config["skip_words"]:
                continue
            neighbours = self.get_neighbours(word, pos, self.config["top_n"])
            for neighbour in neighbours:
                x_new = self.config["processor"].detokenizer(self.do_replace(x_orig, neighbour, counter))
                pred_target = clsf.get_pred([x_new])[0]
                if targeted and pred_target == target:
                    return (x_new, target)
                elif not targeted and pred_target != target:
                    return (x_new, pred_target)
        return None
      
    def do_replace(self, x_cur, word, index):
        ret = x_cur
        ret[index] = word
        return ret
             
    def get_neighbours(self, word, POS, num):
        threshold = self.config["neighbour_threshold"]
        try:
            sub_words = list(
                map(
                    lambda x: x[0],
                    self.config["substitute"](word, pos=POS, threshold=threshold)[:num],
                )
            )
            neighbours = []
            for sub_word in sub_words:
                if self.config["processor"].get_tokens(sub_word)[0][1] == POS:
                    neighbours.append(sub_word)
            return neighbours
        except WordNotInDictionaryException:
            return []