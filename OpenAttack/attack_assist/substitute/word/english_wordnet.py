from .base import WordSubstitute
from ....tags import TAG_English
from ....data_manager import DataManager
from ....exceptions import WordNotInDictionaryException



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


class WordNetSubstitute(WordSubstitute):

    TAGS = { TAG_English }

    def __init__(self, k = None):
        """
        English word substitute based on wordnet.

        Args:
            k: Top-k results to return. If k is `None`, all results will be returned. Default: 50
        
        :Data Requirements: :py:data:`.TProcess.NLTKWordNet`
        :Language: english
        
        """

        self.wn = DataManager.load("TProcess.NLTKWordNet")
        self.k = k

    def substitute(self, word: str, pos: str):
        if pos == "other":
            raise WordNotInDictionaryException()
        pos_in_wordnet = {
            "adv": "r",
            "adj": "a",
            "verb": "v",
            "noun": "n"
        }[pos]

        wordnet_synonyms = []
        synsets = self.wn.synsets(word, pos=pos_in_wordnet)
        for synset in synsets:
            wordnet_synonyms.extend(synset.lemmas())
        synonyms = []
        for wordnet_synonym in wordnet_synonyms:
            spacy_synonym = wordnet_synonym.name().replace('_', ' ').split()[0]
            synonyms.append(spacy_synonym)  # original word
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
        if self.k is not None:
            ret = ret[:self.k]
        return ret
