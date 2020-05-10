from ..text_processor import TextProcessor
from ..data_manager import DataManager

class DefaultTextProcessor(TextProcessor):
    def __init__(self):
        self.nltk = __import__("nltk")
        self.tokenize = None
        self.tag = None  # LazyLoad
        self.lemmatize = None
        self.delemmatize = None
        self.ner = None
        self.parser = None
        self.wordnet = None
        

    def get_tokens(self, sentence):
        if self.tokenize is None:
            self.tokenize = DataManager.load("NLTKSentTokenizer")
        if self.tag is None:
            self.tag = DataManager.load("NLTKPerceptronPosTagger")
        return self.tag(self.tokenize(sentence))

    def get_lemmas(self, token_and_pos):
        if self.lemmatize is None:
            self.lemmatize = DataManager.load("NLTKWordnet").lemma
        if not isinstance(token_and_pos, list):
            return self.lemmatize(token_and_pos[0], token_and_pos[1])
        else:
            return [ self.lemmatize(token, pos) for token, pos in token_and_pos ]

    def get_delemmas(self, lemma_and_pos):
        if self.delemmatize is None:
            self.delemmatize = DataManager.load("NLTKWordnetDelemma")
        if not isinstance(lemma_and_pos, list):
            token, pos = lemma_and_pos
            return self.delemmatize[token][pos] if (token in self.delemmatize) and (pos in self.delemmatize[token]) else token
        else:
            return [
                self.delemmatize[token][pos] 
                    if (token in self.delemmatize) and (pos in self.delemmatize[token]) else token
                    for token, pos in lemma_and_pos
                ]

    def get_ner(self, sentence):
        if self.ner is None:
            self.ner = DataManager.load("StanfordNER")
        if self.tokenize is None:
            self.tokenize = DataManager.load("NLTKSentTokenizer")
        
        ret = []
        if isinstance(sentence, list):  # list of tokens
            tokens = sentence
            nes = self.ner(tokens)

            ne_buffer = []
            ne_start_pos = 0
            ne_last_pos = 0
            last_NE = False
            it = 0
            for word, ne in nes:
                if ne == "O":
                    if last_NE:
                        last_NE = False
                        ret.append((" ".join(ne_buffer), ne_start_pos, ne_last_pos, ne_type))
                else:
                    if (not last_NE) or (ne_type != ne):
                        if last_NE:
                            # append last ne
                            ret.append((" ".join(ne_buffer), ne_start_pos, ne_last_pos, ne_type))
                        # new entity
                        ne_start_pos = it
                        ne_last_pos = it + 1
                        ne_type = ne
                        ne_buffer = [ word ]
                        last_NE = True
                    else:
                        ne_last_pos = it + 1
                        ne_buffer.append( word )
                if last_NE:
                    # handle the last NE
                    ret.append((" ".join(ne_buffer), ne_start_pos, ne_last_pos, ne_type))
                it += 1
        else:
            tokens = self.tokenize(sentence)
            nes = self.ner(tokens)

            ne_buffer = []
            ne_type = ""
            ne_start_pos = 0
            ne_last_pos = 0
            last_NE = False
            it = 0

            for word, ne in nes:
                it += sentence[it:].find(word)

                if ne == "O":
                    if last_NE:
                        last_NE = False
                        ret.append((" ".join(ne_buffer), ne_start_pos, ne_last_pos, ne_type))
                else:
                    if (not last_NE) or (ne_type != ne):
                        if last_NE:
                            # append last ne
                            ret.append((" ".join(ne_buffer), ne_start_pos, ne_last_pos, ne_type))
                        # new entity
                        ne_start_pos = it
                        ne_last_pos = it + len(word)
                        ne_type = ne
                        ne_buffer = [ word ]
                        last_NE = True
                    else:
                        ne_last_pos = it + len(word)
                        ne_buffer.append( word )
                it += len(word)
            if last_NE:
                # handle the last NE
                ret.append((" ".join(ne_buffer), ne_start_pos, ne_last_pos, ne_type))
        return ret


    def get_parser(self, sentence):
        if self.parser is None:
            self.parser = DataManager.load("StanfordParser")
        return list(self.parser(sentence))[0]

    def get_wsd(self, tokens_and_pos):
        if self.wordnet is None:
            self.wordnet = DataManager.load("NLTKWordnet")

        def lesk(sentence, word, pos=None):
            sent = set(sentence)
            synsets = self.wordnet.synsets(word)
            if pos is not None:
                synsets = [ss for ss in synsets if str(ss.pos()) == pos]
            if len(synsets) == 0:
                return None
            _, sense = max(
                (len(sent.intersection(ss.definition().split())), ss) for ss in synsets
            )
            return sense.name()
        
        sentoken = []
        sentence = []
        for word, pos in tokens_and_pos:
            sentoken.append(word)
            pp = "n"
            if pos in ["a", "r", "n", "v", "s"]:
                pp = pos
            else:
                if pos[:2] == "JJ":
                    pp = "a"
                elif pos[:2] == "VB":
                    pp = "v"
                elif pos[:2] == "NN":
                    pp = "n"
                elif pos[:2] == "RB":
                    pp = "r"
                else:
                    pp = None
            sentence.append((word, pp))
        ret = []

        for word, pos in sentence:
            ret.append( lesk(sentoken, word, pos) )
        return ret