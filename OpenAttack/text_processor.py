import abc


class TextProcessor(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def get_tokens(self, sentence):
        pass

    @abc.abstractmethod
    def get_lemmas(self, token_and_pos):
        pass

    @abc.abstractmethod
    def get_delemmas(self, lemma_and_pos):
        pass

    @abc.abstractmethod
    def get_ner(self, sentence):
        pass

    @abc.abstractmethod
    def get_parser(self, sentence):
        pass

    @abc.abstractmethod
    def get_wsd(self, tokens):
        pass
