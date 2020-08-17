"""
:type: OpenAttack.utils.BertClassifier
:Size: 1.23GB
:Package Requirements:
    * transformers
    * pytorch

Pretrained BERT model on SNLI dataset. See :py:data:`Dataset.SNLI` for detail.
"""

from OpenAttack.utils import make_zip_downloader, BertClassifier

NAME = "Victim.BERT.SNLI"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/victim/bert_snli.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    from OpenAttack import Classifier
    return BertClassifier(path, 3)
