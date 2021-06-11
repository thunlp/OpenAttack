"""
:type: OpenAttack.utils.WordVector
:Size: 1.52GB

Word2vec Word Embedding `[page] <https://code.google.com/archive/p/word2vec/>`__
"""
import numpy as np
import os, pickle
from OpenAttack.utils import make_zip_downloader

NAME = "AttackAssist.Word2Vec"

URL = "/TAADToolbox/word2vec.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    from OpenAttack.attack_assist import WordEmbedding
    word2id = pickle.load( open( os.path.join(path, "word2id.pkl"), "rb") )
    wordvec = pickle.load( open( os.path.join(path, "wordvec.pkl"), "rb") )
    return WordEmbedding(word2id, wordvec)
