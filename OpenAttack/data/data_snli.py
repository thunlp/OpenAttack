"""
:type: a tuple of three :py:class:`.Dataset` s, `(train, valid, test)`.
:Size: 72.596MB

SNLI dataset which is used to train victim models.
"""
import pickle

NAME = "Dataset.SNLI"
DOWNLOAD = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/dataset/snli.pkl"


def LOAD(path):
    from OpenAttack.utils import Dataset, DataInstance

    def mapping(data):
        return Dataset([
            DataInstance(
                x=it[0],
                y=it[2],
                meta= { "reference": it[1] }
            ) for it in data
        ], copy=False)

    train, valid, test = pickle.load(open(path, "rb"))
    return mapping(train), mapping(valid), mapping(test)
