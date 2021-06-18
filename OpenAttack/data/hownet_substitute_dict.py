
from OpenAttack.utils import make_zip_downloader
import os

NAME = "AttackAssist.HownetSubstituteDict"

URL = "/TAADToolbox/hownet_candidate.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    return os.path.join(path, "hownet_candidate/hownet_candidate.pkl")
