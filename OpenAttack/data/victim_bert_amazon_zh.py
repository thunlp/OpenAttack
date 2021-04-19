"""
:type: OpenAttack.utils.BertClassifier
:Size: 992.75 MB
:Package Requirements:
    * transformers
    * pytorch

Pretrained BERT model on Amazon Reviews (Chinese) dataset.
"""

from OpenAttack.utils import make_zip_downloader, BertClassifier
import os

NAME = "Victim.BERT.AMAZON_ZH"

URL = "https://cdn.data.thunlp.org/TAADToolbox/victim/bert_amazon_reviews_zh.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    from OpenAttack import Classifier
    return BertClassifier( path, 5)
