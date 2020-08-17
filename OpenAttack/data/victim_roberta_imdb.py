"""
:type: OpenAttack.utils.RobertaClassifier
:Size: 1.18GB
:Package Requirements:
    * transformers
    * pytorch

Pretrained ROBERTA model on IMDB dataset. See :py:data:`Dataset.IMDB` for detail.
"""

from OpenAttack.utils import make_zip_downloader, RobertaClassifier

NAME = "Victim.ROBERTA.IMDB"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/victim/roberta_imdb.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    from OpenAttack import Classifier
    return RobertaClassifier(path, 2)