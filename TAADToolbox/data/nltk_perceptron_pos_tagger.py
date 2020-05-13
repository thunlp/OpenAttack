from TAADToolbox.utils import make_zip_downloader
import os

NAME = "NLTKPerceptronPosTagger"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/averaged_perceptron_tagger.pickle.zip"
DOWNLOAD = make_zip_downloader(URL, "averaged_perceptron_tagger.pickle")


def LOAD(path):
    ret = __import__("nltk").tag.PerceptronTagger(load=False)
    ret.load(os.path.join(path, "averaged_perceptron_tagger.pickle"))
    return ret.tag
