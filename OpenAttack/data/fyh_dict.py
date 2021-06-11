"""
:type: dict
:Size: 127.6KB

Dictionary file for ChineseFYHChar substitute.
"""
import os
import pickle
from OpenAttack.utils import make_zip_downloader

NAME = "AttackAssist.FYH"

URL = "/TAADToolbox/fyh_dict.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    with open(os.path.join(path, 'tra_dict.pkl'), 'rb') as f:
        tra_dict = pickle.load(f)
    with open(os.path.join(path, 'var_dict.pkl'), 'rb') as f:
        var_dict = pickle.load(f)
    with open(os.path.join(path, 'hot_dict.pkl'), 'rb') as f:
        hot_dict = pickle.load(f)

    return tra_dict, var_dict, hot_dict
