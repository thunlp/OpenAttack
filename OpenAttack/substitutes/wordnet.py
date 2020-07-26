from .base import WordSubstitute
from ..data_manager import DataManager
from ..exceptions import UnknownPOSException



def prefilter(token, synonym):  # 预过滤（原词，一个候选词
    if (len(synonym.split()) > 2 or (  # the synonym produced is a phrase
            synonym == token) or (  # the pos of the token synonyms are different
            token == 'be') or (
            token == 'is') or (
            token == 'are') or (
            token == 'am')):  # token is be
        return False
    else:
        return True


def get_pos(pos_tag):
    pos = pos_tag[0]
    if pos == 'a':
        if pos_tag[2] == 'v':
            pos = 'r'
    return pos


class WordNetSubstitute(WordSubstitute):
    """
    :Data Requirements: :py:data:`.TProcess.NLTKWordNet`

    An implementation of :py:class:`.WordSubstitute`.

    WordNet synonym substitute.

    """
    def __init__(self):
        self.wn = DataManager.load("TProcess.NLTKWordNet")

    def __call__(self, word, pos_tag, threshold=None):
        pp = "noun"
        if pos_tag[:2] == "JJ":
            pp = "adj"
        elif pos_tag[:2] == "VB":
            pp = "verb"
        elif pos_tag[:2] == "NN":
            pp = "noun"
        elif pos_tag[:2] == "RB":
            pp = "adv"
        else:
            pp = None
        pos_tag = pp
        if pos_tag is None:
            return [word]
        if pos_tag not in ['noun', 'verb', 'adj', 'adv']:
            raise UnknownPOSException(word, pos_tag)
        pos = get_pos(pos_tag)

        wordnet_synonyms = []
        synsets = self.wn.synsets(word, pos=pos)
        for synset in synsets:
            wordnet_synonyms.extend(synset.lemmas())
        synonyms = []
        for wordnet_synonym in wordnet_synonyms:
            spacy_synonym = wordnet_synonym.name().replace('_', ' ').split()[0]
            synonyms.append(spacy_synonym)  # 原词
        token = word.replace('_', ' ').split()[0]

        sss = []
        for synonym in synonyms:
            if prefilter(token, synonym):
                sss.append(synonym)
        synonyms = sss[:]

        synonyms_1 = []
        for synonym in synonyms:
            if synonym.lower() in synonyms_1:
                continue
            synonyms_1.append(synonym.lower())

        ret = []
        for syn in synonyms_1:
            ret.append((syn, 1))
        return ret
