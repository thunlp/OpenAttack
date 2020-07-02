from TAADToolbox.utils import make_zip_downloader
import os

NAME = "UniversalSentenceEncoder"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/usencoder.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    return os.path.join(path, "usencoder")
