"""
:type: nltk.WordNetCorpusReader
:Size: 10.283MB

Model files for wordnet in nltk.
`[page] <http://wordnet.princeton.edu/>`__
"""
from OpenAttack.utils import make_zip_downloader

NAME = "TProcess.NLTKWordNet"

URL = "/TAADToolbox/wordnet.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    wnc = __import__("nltk").corpus.WordNetCorpusReader(path, None)
    return wnc
