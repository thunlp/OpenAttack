"""
:type: OpenAttack.utils.BertClassifier
:Size: 386.584MB
:Package Requirements:
    * transformers
    * pytorch

Pretrained BERT model on SST-2 dataset. See :py:data:`Dataset.SST` for detail.
"""

from OpenAttack.utils import make_zip_downloader, BertClassifier

NAME = "Victim.BERT.SST"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/victim/bert_sst.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    from OpenAttack import Classifier
    return BertClassifier(path, 2)
