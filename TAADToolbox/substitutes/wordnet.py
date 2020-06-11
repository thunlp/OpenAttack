'''
    由wordnet提供的近义词（初步过滤）。
    进一步工作：加入NE
'''

from ..substitute import Substitute
# from ..exceptions import
import spacy
from functools import partial
from nltk.corpus import wordnet as wn

# nltk.download('wordnet')
# python -m spacy download en_core_web_sm


def prefilter(token, synonym):  # 预过滤（原词，一个候选词
    if (len(synonym.text.split()) > 2 or (  # the synonym produced is a phrase
            synonym.lemma == token.lemma) or (  # token and synonym are the same
            synonym.tag != token.tag) or (  # the pos of the token synonyms are different
            token.text.lower() == 'be')):  # token is be
        return False
    else:
        return True


def get_pos(pos_tag):
    pos = pos_tag[0]
    if pos == 'a':
        if pos_tag[2] == 'v':
            pos = 'r'
    return pos


class WordNetSubstitute(Substitute):

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def __call__(self, word_or_char, pos_tag):
        if pos_tag not in ['noun', 'verb', 'adj', 'adv']:
            print("pos_tag should be ..")
            # raise exception
        pos = get_pos(pos_tag)  # 整理词性

        wordnet_synonyms = []
        synsets = wn.synsets(word_or_char, pos=pos)
        # print("synsets:", synsets)  # wordnet提供近义词
        for synset in synsets:
            wordnet_synonyms.extend(synset.lemmas())
        # print("wordnet_wynonyms:", wordnet_synonyms)  # lemma
        synonyms = []
        for wordnet_synonym in wordnet_synonyms:
            spacy_synonym = self.nlp(wordnet_synonym.name().replace('_', ' '))[0]  # nlp = spacy.load('en_core_web_sm')
            synonyms.append(spacy_synonym)  # 原词
        # print("synonyms:", synonyms)
        token = self.nlp(word_or_char.replace('_', ' '))[0]

        synonyms = filter(partial(prefilter, token), synonyms)  # 初步过滤

        synonyms_1 = []
        for synonym in synonyms:
            if synonym.text.lower() in synonyms_1:
                continue
            synonyms_1.append(synonym.text.lower())

        ret = []
        for syn in synonyms_1:
            ret.append((syn, 1))
        return ret
