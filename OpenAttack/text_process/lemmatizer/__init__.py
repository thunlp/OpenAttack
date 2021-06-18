from .base import Lemmatizer
from .wordnet_lemmatizer import WordnetLemmatimer


def get_default_lemmatizer(lang):
    from ...tags import TAG_English
    if lang == TAG_English:
        return WordnetLemmatimer()
    return WordnetLemmatimer()
