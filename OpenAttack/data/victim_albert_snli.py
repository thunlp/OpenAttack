"""
:type: OpenAttack.utils.AlbertClassifier
:Size: 788.672MB
:Package Requirements:
    * transformers
    * pytorch

Pretrained ALBERT model on SNLI dataset. See :py:data:`Dataset.SNLI` for detail.
"""

from OpenAttack.utils import make_zip_downloader, AlbertClassifier

NAME = "Victim.ALBERT.SNLI"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/victim/albert_snli.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    from OpenAttack import Classifier
    return AlbertClassifier(path, 3)