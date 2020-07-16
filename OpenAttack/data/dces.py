"""
:Size: 632.887KB
:Package Requirements:
    * pickle

Vec-colnames and neighber matrix used in Substitute DECS. See :py:data:`DCES` for detail.
"""

import os
from OpenAttack.utils import make_zip_downloader

NAME = "DCES"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/DCES.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    with open(os.path.join(path, 'vec_colnames.pkl'), 'rb+') as f:
        vec_colnames = __import__("pickle").load(f)
    with open(os.path.join(path, 'descs.pkl'), 'rb+') as f:
        descs = __import__("pickle").load(f)
    with open(os.path.join(path, 'neigh.pkl'), 'rb+') as f:
        neigh = __import__("pickle").load(f)
    return vec_colnames, descs, neigh
