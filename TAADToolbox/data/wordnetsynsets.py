NAME = "WordnetSynsets"
DOWNLOAD = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/wordnetsynsets.txt"

def LOAD(path):
    wnc = __import__("nltk").corpus.wordnet
    return wnc
