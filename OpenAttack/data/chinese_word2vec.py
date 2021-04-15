"""
:type: OpenAttack.utils.WordVector
:Size: 3GB
"""
import numpy as np
import os
from OpenAttack.utils import WordVector, make_zip_downloader

NAME = "AttackAssist.ChineseWord2Vec"

URL = "https://cdn.data.thunlp.org/TAADToolbox/chinese-merge-word-embedding.txt.zip"
DOWNLOAD = make_zip_downloader(URL, "chinese-merge-word-embedding.txt")


def LOAD(path):
    with open(os.path.join(path, "chinese-merge-word-embedding.txt"), "r", encoding="utf-8") as f:
        id2vec = []
        word2id = {}
        # f.readline()
        for line in f.readlines():
            tmp = line.strip().split(' ')
            word = tmp[0]
            embed = np.array([float(x) for x in tmp[1:]])
            if len(embed) != 300:
                continue
            word2id[word] = len(word2id)
            id2vec.append(embed)
        id2vec = np.stack(id2vec)
    return WordVector(word2id, id2vec)
