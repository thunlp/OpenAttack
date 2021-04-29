"""
:type: nltk.WordNetCorpusReader
:Size: 10.283MB

Model files for wordnet in nltk.
`[page] <http://wordnet.princeton.edu/>`__
"""
from OpenAttack.utils import make_zip_downloader

NAME = "TProcess.NLTKWordNet"

URL = "https://cdn.data.thunlp.org/TAADToolbox/wordnet.zip"
DOWNLOAD = make_zip_downloader(URL)

class Lemmatizer:
    def __init__(self, wnc):
        self.__wnc = wnc
    
    def __call__(self, word, pos):
        pp = "n"
        if pos in ["a", "r", "n", "v", "s"]:
            pp = pos
        else:
            if pos[:2] == "JJ":
                pp = "a"
            elif pos[:2] == "VB":
                pp = "v"
            elif pos[:2] == "NN":
                pp = "n"
            elif pos[:2] == "RB":
                pp = "r"
            else:
                pp = None
        if pp is None:  # do not need lemmatization
            return word
        lemmas = self.__wnc._morphy(word, pp)
        return min(lemmas, key=len) if len(lemmas) > 0 else word

def LOAD(path):
    wnc = __import__("nltk").corpus.WordNetCorpusReader(path, None)
    wnc.lemma = Lemmatizer(wnc)
    return wnc
