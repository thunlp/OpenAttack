"""
:type: dict
:Size: 150.386MB

Models used in SCPNAttacker.
`[page] <https://github.com/miyyer/scpn>`__
"""
from OpenAttack.utils import make_zip_downloader
import os

NAME = "AttackAssist.SCPN"

URL = "/TAADToolbox/scpn.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    flist = ["scpn.pt", "parse_generator.pt", "parse_vocab.pkl", "bpe.codes", "vocab.txt", "ptb_tagset.pkl"]
    return {
        it: os.path.join(path, it) for it in flist
    }
