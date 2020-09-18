"""
:type: OpenAttack.utils.XlnetClassifier
:Size: 1.25GB
:Package Requirements:
    * transformers
    * pytorch

Pretrained XLNET model on SNLI dataset. See :py:data:`Dataset.SNLI` for detail.
"""

from OpenAttack.utils import make_zip_downloader, XlnetClassifier

NAME = "Victim.XLNET.SNLI"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/victim/xlnet_snli.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    from OpenAttack import Classifier
    return XlnetClassifier(path, 3)