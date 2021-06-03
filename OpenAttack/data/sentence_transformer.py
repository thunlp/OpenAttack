"""
:type: str
:Size: 916.57MB

Model files for Universal Sentence Encoder in tensorflow_hub.
`[pdf] <https://arxiv.org/pdf/1803.11175>`__
`[page] <https://tfhub.dev/google/universal-sentence-encoder/4>`__
"""
from OpenAttack.utils import make_zip_downloader
import os

NAME = "AttackAssist.SentenceTransformer"

URL = "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/stsb-bert-large.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    return os.path.join(path, "stsb-bert-large")
