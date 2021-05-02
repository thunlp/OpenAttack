from ..text_processor import TextProcessor

class ChineseTextProcessor(TextProcessor):
    """
    An implementation of :class:`OpenAttack.TextProcessor` mainly uses ``thulac`` toolkit.
    """

    def __init__(self):
        self.__tokenize = None
        self.__tag = None  # LazyLoad
        self.__lemmatize = None
        self.__delemmatize = None
        self.__ner = None
        self.__parser = None
        self.__wordnet = None

        import jieba
        import jieba.posseg as pseg
        self.__tokenize = pseg.cut
        jieba.initialize()

    def get_tokens(self, sentence):
        """
        :param str sentence: A sentence that we want to tokenize.
        :return: The result is a list of tuples, *(word, pos_tag)*.
        :rtype: list of tuples
        """
        mapping = {
            'v': 'VBD',
            'n': 'NN',
            'r': 'PRP',
            't': 'NN',
            'm': 'DT',
            'f': 'IN',
            'a': 'JJ',
            'd': 'RB'
        }


        ans = []
        for pair in self.__tokenize(sentence):  
            if pair.flag[0] in mapping:
                ans.append((pair.word, mapping[ pair.flag[0] ]))
            else:
                ans.append((pair.word, "OTHER" ))
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
        return ''.join(tokens)