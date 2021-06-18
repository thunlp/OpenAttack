from .base import Tokenizer
from ...data_manager import DataManager
from ...tags import *


_POS_MAPPING = {
    "v": "verb",
    "n": "noun",
    "t": "noun",
    "a": "adj",
    "d": "adv"
}
class JiebaTokenizer(Tokenizer):
    """
    Tokenizer based on jieba.posseg

    :Package Requirements:
        * jieba
    :Language: chinese
    """

    TAGS = { TAG_Chinese }

    def __init__(self) -> None:
        import jieba
        import jieba.posseg as pseg
        self.__tokenize = pseg.cut
        jieba.initialize()
    
    def do_tokenize(self, x, pos_tagging):
        ret = []
        for pair in self.__tokenize(x):
            if pos_tagging:
                pos = pair.flag[0]
                if pos in _POS_MAPPING:
                    pos = _POS_MAPPING[pos]
                else:
                    pos = "other"
                ret.append( (pair.word, pos) )
            else:
                ret.append( pair.word )
        return ret
    
    def do_detokenize(self, x):
        return "".join(x)
    