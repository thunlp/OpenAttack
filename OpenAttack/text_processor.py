import abc


class TextProcessor(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def get_tokens(self, sentence):
        """
        :param str sentence: A sentence which needs to be tokenized.
        :return: The result is a list of tuples, *(token, POS)*.
        :rtype: list of tuple

        It is recommended to use tags in `"Penn tag set" <https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html>`_ which is supported by all attackers.
        """
        pass

    @abc.abstractmethod
    def get_lemmas(self, token_and_pos):
        """
        :param token_and_pos: A tuple or a list of tuples,  *(token, POS)*.
        :type token_and_pos: list or tuple
        :return: A lemma or a list of lemmas depends on your input.
        :rtype: list or str
        """
        pass

    @abc.abstractmethod
    def get_delemmas(self, lemma_and_pos):
        """
        :param lemma_and_pos: A tuple or a list of tuples, *(lemma, POS)*.
        :type lemma_and_pos: list or tuple
        :return: A word or a list of words, each word represents the specific form of input lemma.
        :rtype: list or str
        """
        pass

    @abc.abstractmethod
    def get_ner(self, sentence):
        """
        :param str sentence: A sentence that we want to extract named entities.
        :return: A list of tuples, *(entity, start, end, label)*
        :rtype: list
        """
        pass

    @abc.abstractmethod
    def get_parser(self, sentence):
        """
        :param str sentence: A setence needs to be parsed
        :return: The result tree of lexicalized parser in string format.
        :rtype: str
        """
        pass

    @abc.abstractmethod
    def get_wsd(self, tokens_and_pos):
        """
        :param list tokens_and_pos: A list of tuples, *(token, POS)*.
        :return: A list of str, represents the sense of each input token.
        :rtype: list
        """
        pass

    @abc.abstractmethod
    def detokenizer(self, tokens):
        """
        :param list tokens: A list of tokens
        :return: A detokenized sentence.
        :rtype: str

        """
        pass
