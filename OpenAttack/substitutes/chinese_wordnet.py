from .base import WordSubstitute
from ..exceptions import UnknownPOSException

# import nltk
# nltk.download('omw')


def get_pos(pos_tag):
    pos = pos_tag[0]
    if pos == 'a':
        if pos_tag[2] == 'v':
            pos = 'r'
    return pos


class ChineseWordNetSubstitute(WordSubstitute):
    """
    :Data Requirements: :py:data:`.TProcess.NLTKWordNet`

    An implementation of :py:class:`.WordSubstitute`.

    ChineseWordNet synonym substitute.

    """
    def __init__(self):
        super().__init__()

    def __call__(self, word, pos_tag, threshold=None):
        """
        :param word: the raw word; pos_tag: part of speech of the word, threshold: return top k words.
        :return: The result is a list of tuples, *(substitute, 1)*.
        :rtype: list of tuple
        """
        from nltk.corpus import wordnet as wn
        if pos_tag is None:
            pp = None
        elif pos_tag[:2] == "JJ":
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
            return [(word, 1)]
        pos = get_pos(pos_tag)

        pos_list = ['noun', 'verb', 'adj', 'adv']
        pos_set = set(pos_list)
        if pos_tag not in pos_set:
            raise UnknownPOSException(word, pos_tag)

        synonyms = []
        for synset in wn.synsets(word, pos=pos, lang='cmn'):
            for lemma in synset.lemma_names('cmn'):
                if lemma == word:
                    continue
                synonyms.append((lemma, 1))
        if threshold:
            return synonyms[:threshold]
        return synonyms
