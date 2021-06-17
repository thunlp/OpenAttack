from typing import List, Optional
from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...text_process.tokenizer import Tokenizer, get_default_tokenizer
from ...attack_assist.substitute.word import WordSubstitute, get_default_substitute
from ...utils import get_language, check_language, language_by_name
from ...exceptions import WordNotInDictionaryException
from ...tags import Tag
from ...attack_assist.filter_words import get_default_filter_words

class HotFlipAttacker(ClassificationAttacker):
    @property
    def TAGS(self):
        return { self.__lang_tag, Tag("get_pred", "victim"), Tag("get_prob", "victim") }

    def __init__(self,
            substitute : Optional[WordSubstitute] = None,
            tokenizer : Optional[Tokenizer] = None,
            filter_words : List[str] = None,
            lang = None
        ):
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

        lst = []
        if tokenizer is not None:
            lst.append(tokenizer)
        if substitute is not None:
            lst.append(substitute)
        if len(lst) > 0:
            self.__lang_tag = get_language(lst)
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

    def attack(self, victim: Classifier, sentence : str, goal: ClassifierGoal):
        x_orig = sentence.lower()

        x_orig = self.tokenizer.tokenize(x_orig)
        x_pos =  list(map(lambda x: x[1], x_orig))
        x_orig = list(map(lambda x: x[0], x_orig))
        
        counter = -1
        for word, pos in zip(x_orig, x_pos):
            counter += 1
            if word in self.filter_words:
                continue
            neighbours = self.get_neighbours(word, pos)
            for neighbour in neighbours:
                x_new = self.tokenizer.detokenize(self.do_replace(x_orig, neighbour, counter))
                pred_target = victim.get_pred([x_new])[0]
                if goal.check(x_new, pred_target):
                    return x_new
        return None
      
    def do_replace(self, x_cur, word, index):
        ret = x_cur
        ret[index] = word
        return ret
             
    def get_neighbours(self, word, POS):
        try:
            return list( map(lambda x: x[0], self.substitute(word, POS)) )
        except WordNotInDictionaryException:
            return []