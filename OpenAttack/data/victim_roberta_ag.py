"""
:type: OpenAttack.utils.RobertaClassifier
:Size: 1.22GB
:Package Requirements:
    * transformers
    * pytorch

Pretrained ROBERTA model on AG-4 dataset. See :py:data:`Dataset.AG` for detail.
"""

from OpenAttack.utils import make_zip_downloader, RobertaClassifier

NAME = "Victim.ROBERTA.AG"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/victim/roberta_ag.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    from OpenAttack import Classifier
    return RobertaClassifier(path, 5)