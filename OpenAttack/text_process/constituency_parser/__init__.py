from .base import ConstituencyParser
from .stanford_parser import StanfordParser

def get_default_constituency_parser(lang):
    from ...tags import TAG_English
    if lang == TAG_English:
        return StanfordParser()
    return StanfordParser()
