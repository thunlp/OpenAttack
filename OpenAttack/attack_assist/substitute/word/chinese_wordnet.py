from .base import WordSubstitute
from ....exceptions import UnknownPOSException
from ....tags import *


class ChineseWordNetSubstitute(WordSubstitute):

    TAGS = { TAG_Chinese }

    """
    :Data Requirements: :py:data:`.TProcess.NLTKWordNet`

    An implementation of :py:class:`.WordSubstitute`.

    ChineseWordNet synonym substitute.

    """
    def __init__(self, k = None):
        self.k = k
    
    def substitute(self, word: str, pos: str):
        from nltk.corpus import wordnet as wn
        
        pos_in_wordnet = {
            "adv": "r",
            "adj": "a",
            "verb": "v",
            "noun": "n"
        }[pos]

        synonyms = []
        for synset in wn.synsets(word, pos=pos_in_wordnet, lang='cmn'):
            for lemma in synset.lemma_names('cmn'):
                if lemma == word:
                    continue
                synonyms.append((lemma, 1))
        
        if self.k is not None:
            return synonyms[:self.k]

        return synonyms
