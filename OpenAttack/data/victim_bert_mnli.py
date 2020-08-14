"""
:type: OpenAttack.utils.BertClassifier
:Size: 1.23GB
:Package Requirements:
    * transformers
    * pytorch

Pretrained BERT model on MNLI dataset. See :py:data:`Dataset.MNLI` for detail.
"""

from OpenAttack.utils import make_zip_downloader, BertClassifier

NAME = "Victim.BERT.MNLI"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/victim/bert_mnli.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    from OpenAttack import Classifier
    return BertClassifier(path, 2)
