"""
:type: function
:Size: 158.351KB

Model files for nltk punkt sentence tokenizer.
"""
from OpenAttack.utils import make_zip_downloader
import os

NAME = "TProcess.NLTKSentTokenizer"

URL = "/TAADToolbox/punkt.english.pickle.zip"
DOWNLOAD = make_zip_downloader(URL, "english.pickle")


def LOAD(path):
    return __import__("nltk").data.load("file:" + os.path.join(path, "english.pickle")).tokenize

