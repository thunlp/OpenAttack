"""
:type: OpenAttack.utils.WordVector
:Size: 2.273GB

GloVe Word Embedding `[page] <https://nlp.stanford.edu/projects/glove/>`__
"""
import numpy as np
import os, pickle
from OpenAttack.utils import make_zip_downloader

NAME = "AttackAssist.GloVe"

URL = "/TAADToolbox/glove.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    from OpenAttack.attack_assist import WordEmbedding
    word2id = pickle.load( open( os.path.join(path, "word2id.pkl"), "rb") )
    wordvec = np.load( os.path.join(path, "wordvec.npy") )
    return WordEmbedding(word2id, wordvec)
