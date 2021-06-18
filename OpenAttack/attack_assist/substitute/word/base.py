from typing import List, Optional, Tuple
from ....exceptions import UnknownPOSException, WordNotInDictionaryException

POS_LIST = ["adv", "adj", "noun", "verb", "other"]

class WordSubstitute(object):
    def __call__(self, word : str, pos : Optional[str] = None) -> List[Tuple[str, float]]:
        """
        In WordSubstitute, we return a list of words that are semantically similar to the input word.
        
        Args:
            word: A single word.
            pos: POS tag of input word. Must be one of the following: ``["adv", "adj", "noun", "verb", "other", None]``
        
        Returns:
            A list of words and their distance to original word (distance is a number between 0 and 1, with smaller indicating more similarity)
        Raises:
            WordNotInDictionaryException: input word not in the dictionary of substitute algorithm
            UnknownPOSException: invalid pos tagging

        """
        
        if pos is None:
            ret = {}
            for sub_pos in POS_LIST:
                try:
                    for word, sim in self.substitute(word, sub_pos):
                        if word not in ret:
                            ret[word] = sim
                        else:
                            ret[word] = max(ret[word], sim)
                except WordNotInDictionaryException:
                    continue
            list_ret = []
            for word, sim in ret.items():
                list_ret.append((word, sim))
            if len(list_ret) == 0:
                raise WordNotInDictionaryException()
            return sorted( list_ret, key=lambda x: -x[1] )
        elif pos not in POS_LIST:
            raise UnknownPOSException("Invalid `pos` %s (expect %s)" % (pos, POS_LIST) )
        return self.substitute(word, pos)
    
    def substitute(self, word : str, pos : str) -> List[Tuple[str, float]]:
        raise NotImplementedError()
        