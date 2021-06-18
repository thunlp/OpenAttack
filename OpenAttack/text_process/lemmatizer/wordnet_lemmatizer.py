from .base import Lemmatizer
from ...tags import *
from ...data_manager import DataManager

POS_MAPPING = {
    "adv": "r",
    "adj": "a",
    "verb": "v",
    "noun": "n"
}

_DELEMMA_POS_MAPPING = {
    "JJ": "adj",
    "VB": "verb",
    "NN": "noun",
    "RB": "adv"
}

class WordnetLemmatimer(Lemmatizer):
    """
    Lemmatizer based on nltk.wordnet
    
    :Language: english
    """

    TAGS = { TAG_English }

    def __init__(self) -> None:
        self.wnc = DataManager.load("TProcess.NLTKWordNet")
        old_delema = DataManager.load("TProcess.NLTKWordNetDelemma")
        self.__delema = {}
        for word in old_delema.keys():
            self.__delema[word] = {}
            for kw, val in old_delema[word].items():
                if kw[:2] in _DELEMMA_POS_MAPPING:
                    pos = _DELEMMA_POS_MAPPING[kw[:2]]
                    self.__delema[word][pos] = val
        
    def do_lemmatize(self, token, pos):
        if pos not in POS_MAPPING:
            return token
        pos_in_wordnet = POS_MAPPING[pos]

        lemmas = self.wnc._morphy(token, pos_in_wordnet)
        return min(lemmas, key=len) if len(lemmas) > 0 else token

    
    def do_delemmatize(self, lemma, pos):
        if (lemma in self.__delema) and (pos in self.__delema[lemma]):
            return self.__delema[lemma][pos]
        return lemma