"""
:type: OpenAttack.utils.AlbertClassifier
:Size: 788.668MB
:Package Requirements:
    * transformers
    * pytorch

Pretrained ALBERT model on MNLI dataset. See :py:data:`Dataset.MNLI` for detail.
"""

from OpenAttack.utils import make_zip_downloader, AlbertClassifier

NAME = "Victim.ALBERT.MNLI"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/victim/albert_mnli.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    from OpenAttack import Classifier
    return AlbertClassifier(path, 2)