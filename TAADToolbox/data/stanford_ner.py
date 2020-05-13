from TAADToolbox.utils import make_zip_downloader
import os

NAME = "StanfordNER"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/stanford_ner_small.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    return (
        __import__("nltk")
        .StanfordNERTagger(
            model_filename=os.path.join(path, "english.muc.7class.distsim.crf.ser.gz"),
            path_to_jar=os.path.join(path, "stanford-ner.jar"),
        )
        .tag
    )
