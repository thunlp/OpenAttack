import os
from TAADToolbox.utils import make_zip_downloader

NAME = "SpacyEn"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/spacyen.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    with open(os.path.join(path, "spacyen.pkl"), "rb+") as f:
        nlp = __import__("pickle").load(f)
    return nlp
