"""
:type: OpenAttack.utils.AlbertClassifier
:Size: 788.662MB
:Package Requirements:
    * transformers
    * pytorch

Pretrained ALBERT model on IMDB dataset. See :py:data:`Dataset.IMDB` for detail.
"""

from OpenAttack.utils import make_zip_downloader, AlbertClassifier

NAME = "Victim.ALBERT.IMDB"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/victim/albert_imdb.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    from OpenAttack import Classifier
    return AlbertClassifier(path, 2)