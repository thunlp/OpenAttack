"""
:type: OpenAttack.utils.AlbertClassifier
:Size: 788.697MB
:Package Requirements:
    * transformers
    * pytorch

Pretrained ALBERT model on AG-4 dataset. See :py:data:`Dataset.AG` for detail.
"""

from OpenAttack.utils import make_zip_downloader, AlbertClassifier

NAME = "Victim.ALBERT.AG"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/victim/albert_ag.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    from OpenAttack import Classifier
    return AlbertClassifier(path, 5)