import os
from TAADToolbox.utils import make_zip_downloader

NAME = "TREEBANK"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/treebank.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    with open(os.path.join(path, "treebank.pkl"), "rb+") as f:
        nlp = __import__("pickle").load(f)
    return nlp
