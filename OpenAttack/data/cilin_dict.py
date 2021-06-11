"""
:type: dict
:Size: 1357KB

Dictionary file for ChineseCiLin substitute.
"""
import os
import pickle
from OpenAttack.utils import make_zip_downloader

NAME = "AttackAssist.CiLin"

URL = "/TAADToolbox/cilin_dict.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    with open(os.path.join(path, 'cilin_dict.pkl'), 'rb') as f:
        cilin_dict = pickle.load(f)

    return cilin_dict
