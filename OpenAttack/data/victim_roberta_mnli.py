"""
:type: OpenAttack.utils.RobertaClassifier
:Size: 1.22GB
:Package Requirements:
    * transformers
    * pytorch

Pretrained ROBERTA model on MNLI dataset. See :py:data:`Dataset.MNLI` for detail.
"""

from OpenAttack.utils import make_zip_downloader, RobertaClassifier

NAME = "Victim.ROBERTA.MNLI"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/victim/roberta_mnli.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    from OpenAttack import Classifier
    return RobertaClassifier(path, 3)