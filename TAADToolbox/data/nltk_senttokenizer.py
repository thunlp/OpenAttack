from TAADToolbox.utils import make_zip_downloader
import os

NAME = "NLTKSentTokenizer"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/punkt.english.pickle.zip"
DOWNLOAD = make_zip_downloader(URL, "english.pickle")

def LOAD(path):
    sent_tokenizer = __import__("nltk").data.load(os.path.join(path, "english.pickle")).tokenize
    word_tokenizer = __import__("nltk").NLTKWordTokenizer().tokenize
    def tokenize(sent):
        sentences = sent_tokenizer(sent)
        return [ token for sent in sentences for token in word_tokenizer(sent) ]
    return tokenize