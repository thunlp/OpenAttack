from OpenAttack.utils import make_zip_downloader
import os

NAME = "TranslationModels"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/translation_models.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    flist = ["english_french_model_acc_71.05_ppl_3.71_e13.pt", "english_portuguese_model_acc_70.75_ppl_4.32_e13.pt", 
            "french_english_model_acc_68.51_ppl_4.43_e13.pt", "portuguese_english_model_acc_69.93_ppl_5.04_e13.pt"]
    return {
        it: os.path.join(path, it) for it in flist
    }
