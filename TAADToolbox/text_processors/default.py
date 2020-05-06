from ..text_processor import TextProcessor


class DefaultTextProcessor(TextProcessor):
    def __init__(self):
        self.nltk = __import__("nltk")

    def get_tokens(self, sentence):
        return self.nltk.pos_tag(self.nltk.tokenize.word_tokenize(sentence))

    def get_lemmas(self, token_and_pos):
        raise NotImplementedError

    def get_delemmas(self, lemma_and_pos):
        raise NotImplementedError

    def get_ner(self, sentence):
        raise NotImplementedError

    def get_parser(self, sentence):
        raise NotImplementedError

    def get_wsd(self, tokens):
        raise NotImplementedError
