"""
:type: OpenAttack.utils.AlbertClassifier
:Size: 788.66MB
:Package Requirements:
    * transformers
    * pytorch

Pretrained ALBERT model on SST-2 dataset. See :py:data:`Dataset.SST` for detail.
"""

from OpenAttack.utils import make_zip_downloader, AlbertClassifier

NAME = "Victim.ALBERT.SST"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/victim/albert_sst.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    from OpenAttack import Classifier
    return AlbertClassifier(path, 2)