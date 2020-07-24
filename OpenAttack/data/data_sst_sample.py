"""
:type: Dataset
:Size: 112.519KB

A subset of :py:data:`.Dataset.SST`, used to evaluate attackers and classifiers.
"""
import pickle

NAME = "Dataset.SST.sample"
DOWNLOAD = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/dataset/SST.sample.pkl"

def LOAD(path):
    data = pickle.load(open(path, "rb"))
    from OpenAttack.utils import Dataset
    return Dataset(data)