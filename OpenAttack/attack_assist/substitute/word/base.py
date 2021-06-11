from typing import List, Optional, Tuple
from ....exceptions import UnknownPOSException

POS_LIST = ["adv", "adj", "noun", "verb", "other"]

class WordSubstitute(object):
    def __call__(self, word : str, pos : Optional[str] = None):
        """
        :param word: the raw word; 
        :param pos: part of speech of the word (`adj`, `adv`, `noun`, `verb`).
        :return: The result is a list of tuples, *(substitute, distance)*.
        :rtype: list of tuple

        In WordSubstitute, we return a list of words that are semantically similar to the original word.
        """
        if pos is None:
            ret = {}
            for sub_pos in POS_LIST:
                for word, sim in self.substitute(word, sub_pos):
                    if word not in ret:
                        ret[word] = sim
                    else:
                        ret[word] = max(ret[word], sim)
            list_ret = []
            for word, sim in ret.items():
                list_ret.append((word, sim))
            return sorted( list_ret, key=lambda x: -x[1] )
        elif pos not in POS_LIST:
            raise UnknownPOSException("Invalid `pos` %s (expect %s)" % (pos, POS_LIST) )
        return self.substitute(word, pos)
    
    def substitute(self, word : str, pos : str) -> List[Tuple[str, float]]:
        raise NotImplementedError()
        