"""
:type: OpenAttack.utils.XlnetClassifier
:Size: 1.25GB
:Package Requirements:
    * transformers
    * pytorch

Pretrained XLNET model on AG-4 dataset. See :py:data:`Dataset.AG` for detail.
"""

from OpenAttack.utils import make_zip_downloader, XlnetClassifier

NAME = "Victim.XLNET.AG"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/victim/xlnet_ag.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    from OpenAttack import Classifier
    return XlnetClassifier(path, 4)