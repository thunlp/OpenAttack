import os
from TAADToolbox.utils import make_zip_downloader

NAME = "SpacyECW"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/spacy_en_core_web.pkl.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    with open(os.path.join(path, "spacy_en_core_web.pkl"), "rb+") as f:
        nlp = __import__("pickle").load(f)
    return nlp
