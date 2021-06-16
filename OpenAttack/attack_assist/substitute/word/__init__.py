from .base import WordSubstitute
from .chinese_cilin import ChineseCiLinSubstitute
from .chinese_hownet import ChineseHowNetSubstitute
from .chinese_wordnet import ChineseWordNetSubstitute
from .chinese_word2vec import ChineseWord2VecSubstitute

from .embed_based import EmbedBasedSubstitute

from .english_hownet import HowNetSubstitute
from .english_wordnet import WordNetSubstitute
from .english_counterfit import CounterFittedSubstitute
from .english_word2vec import Word2VecSubstitute
from .english_glove import GloveSubstitute


def get_default_substitute(lang):
    from ....tags import TAG_Chinese, TAG_English
    if lang == TAG_English:
        return WordNetSubstitute()
    if lang == TAG_Chinese:
        return ChineseWordNetSubstitute()
    return WordNetSubstitute()