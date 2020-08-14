"""
:type: OpenAttack.utils.RobertaClassifier
:Size: 1.2GB
:Package Requirements:
    * transformers
    * pytorch

Pretrained ROBERTA model on SNLI dataset. See :py:data:`Dataset.SNLI` for detail.
"""

from OpenAttack.utils import make_zip_downloader, RobertaClassifier

NAME = "Victim.ROBERTA.SNLI"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/victim/roberta_snli.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    from OpenAttack import Classifier
    return RobertaClassifier(path, 3)