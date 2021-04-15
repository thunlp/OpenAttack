from ..text_processor import TextProcessor
from ..data_manager import DataManager


class ChineseTextProcessor(TextProcessor):
    """
    An implementation of :class:`OpenAttack.TextProcessor` mainly uses ``nltk`` toolkit.
    """

    def __init__(self):
        self.nltk = __import__("nltk")
        self.__tokenize = None
        self.__tag = None  # LazyLoad
        self.__lemmatize = None
        self.__delemmatize = None
        self.__ner = None
        self.__parser = None
        self.__wordnet = None

    def get_tokens(self, sentence):
        """
        method: thulac
        """
        dict = {
            'v': 'VBD',
            'n': 'NN',
            'r': 'PRP',
            't': 'NN',
            'm': 'DT',
            'f': 'IN',
            'a': 'JJ',
            'd': 'RB'
        }
        import thulac
        thu = thulac.thulac()
        lists = thu.cut(sentence)
        ans = []
        for i in range(len(lists)):
            if lists[i][1] in dict:
                lists[i][1] = dict[lists[i][1]]
            else:
                lists[i][1] = 'NONE'
            ans.append(tuple(lists[i]))
        return ans

    def get_lemmas(self, token_and_pos):
        """
        :param token_and_pos: A tuple or a list of tuples,  *(token, POS)*.
        :type token_and_pos: list or tuple
        :return: A lemma or a list of lemmas depends on your input.
        :rtype: list or str
        """
        pass
    
    def get_delemmas(self, lemma_and_pos):
        """
        :param lemma_and_pos: A tuple or a list of tuples, *(lemma, POS)*.
        :type lemma_and_pos: list or tuple
        :return: A word or a list of words, each word represents the specific form of input lemma.
        :rtype: list or str
        """
        pass

    def get_ner(self, sentence):
        """
        :param str sentence: A sentence that we want to extract named entities.
        :return: A list of tuples, *(entity, start, end, label)*
        :rtype: list
        """
        pass

    def get_parser(self, sentence):
        """
        :param str sentence: A setence needs to be parsed
        :return: The result tree of lexicalized parser in string format.
        :rtype: str
        """
        pass

    def get_wsd(self, tokens_and_pos):
        """
        :param list tokens_and_pos: A list of tuples, *(token, POS)*.
        :return: A list of str, represents the sense of each input token.
        :rtype: list
        """
        pass

    def detokenizer(self, tokens):
        """
        :param list tokens: A list of tokens
        :return: A detokenized sentence.
        :rtype: str

        """
        return ' '.join(tokens)