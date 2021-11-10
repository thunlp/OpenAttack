"""
:type: OpenAttack.utils.WordVector
:Size: 3GB
"""
import numpy as np
import os
from OpenAttack.utils import make_zip_downloader

NAME = "AttackAssist.ChineseWord2Vec"

URL = "/TAADToolbox/chinese-merge-word-embedding.txt.zip"
DOWNLOAD = make_zip_downloader(URL, "chinese-merge-word-embedding.txt")


def LOAD(path):
    from OpenAttack.attack_assist import WordEmbedding
    with open(os.path.join(path, "chinese-merge-word-embedding.txt"), "r", encoding="utf-8") as f:
        id2vec = []
        word2id = {}
        # f.readline()
        for idx, line in enumerate(f.readlines()):
            tmp = line.strip().split(' ')
            word = tmp[0]
            embed = np.array([float(x) for x in tmp[1:]])
            if len(embed) != 300:
                continue
            word2id[word] = idx
            id2vec.append(embed)
        id2vec = np.stack(id2vec)
    return WordEmbedding(word2id, id2vec)
