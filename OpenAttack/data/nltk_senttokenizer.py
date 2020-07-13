from OpenAttack.utils import make_zip_downloader
import os

NAME = "NLTKSentTokenizer"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/punkt.english.pickle.zip"
DOWNLOAD = make_zip_downloader(URL, "english.pickle")


def LOAD(path):
    return __import__("nltk").data.load(os.path.join(path, "english.pickle")).tokenize

