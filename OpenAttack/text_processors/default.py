from ..text_processor import TextProcessor
from ..data_manager import DataManager




class DefaultTextProcessor(TextProcessor):
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
    
    def __make_tokenizer(self, sent_tokenizer):
        word_tokenizer = __import__("nltk").WordPunctTokenizer().tokenize
        def tokenize(sent):
            sentences = sent_tokenizer(sent)
            return [token for sent in sentences for token in word_tokenizer(sent)]
        return tokenize

    def get_tokens(self, sentence):
        """
        :Data Requirements: :py:data:`TProcess.NLTKSentTokenizer` , :py:data:`.TProcess.NLTKPerceptronPosTagger`

        This method uses ``nltk.WordPunctTokenizer()`` for word tokenization and 
        ``nltk.tag.PerceptronTagger()`` to generate POS tag in "Penn tagset".
        """
        if self.__tokenize is None:
            self.__tokenize = self.__make_tokenizer( DataManager.load("TProcess.NLTKSentTokenizer") )
        if self.__tag is None:
            self.__tag = DataManager.load("TProcess.NLTKPerceptronPosTagger")
        return self.__tag(self.__tokenize(sentence))

    def get_lemmas(self, token_and_pos):
        """
        :Data Requirements: :py:data:`.TProcess.NLTKWordNet`

        This method uses ``nltk.WordNetLemmatier`` to lemmatize tokens.
        """
        if self.__lemmatize is None:
            self.__lemmatize = DataManager.load("TProcess.NLTKWordNet").lemma
        if not isinstance(token_and_pos, list):
            return self.__lemmatize(token_and_pos[0], token_and_pos[1])
        else:
            return [self.__lemmatize(token, pos) for token, pos in token_and_pos]

    def get_delemmas(self, lemma_and_pos):
        """
        :Data Requirements: :py:data:`.TProcess.NLTKWordNetDelemma`
        
        This method uses a pre-processed dict which maps (lemma, pos) to original token for delemmatizing.
        """
        if self.__delemmatize is None:
            self.__delemmatize = DataManager.load("TProcess.NLTKWordNetDelemma")
        if not isinstance(lemma_and_pos, list):
            token, pos = lemma_and_pos
            return (
                self.__delemmatize[token][pos]
                if (token in self.__delemmatize) and (pos in self.__delemmatize[token])
                else token
            )
        else:
            return [
                self.__delemmatize[token][pos]
                if (token in self.__delemmatize) and (pos in self.__delemmatize[token])
                else token
                for token, pos in lemma_and_pos
            ]

    def get_ner(self, sentence):
        """
        :Data Requirements: :py:data:`.TProcess.StanfordNER` , :py:data:`.TProcess.NLTKSentTokenizer`
        :Package Requirements: * **Java**

        This method uses NLTK tokenizer and Stanford NER toolkit which requires Java installed.
        """
        if self.__ner is None:
            self.__ner = DataManager.load("TProcess.StanfordNER")
        if self.__tokenize is None:
            self.__tokenize = DataManager.load("TProcess.NLTKSentTokenizer")

        ret = []
        if isinstance(sentence, list):  # list of tokens
            tokens = sentence
            nes = self.__ner(tokens)

            ne_buffer = []
            ne_start_pos = 0
            ne_last_pos = 0
            last_NE = False
            it = 0
            for word, ne in nes:
                if ne == "O":
                    if last_NE:
                        last_NE = False
                        ret.append(
                            (" ".join(ne_buffer), ne_start_pos, ne_last_pos, ne_type)
                        )
                else:
                    if (not last_NE) or (ne_type != ne):
                        if last_NE:
                            # append last ne
                            ret.append(
                                (
                                    " ".join(ne_buffer),
                                    ne_start_pos,
                                    ne_last_pos,
                                    ne_type,
                                )
                            )
                        # new entity
                        ne_start_pos = it
                        ne_last_pos = it + 1
                        ne_type = ne
                        ne_buffer = [word]
                        last_NE = True
                    else:
                        ne_last_pos = it + 1
                        ne_buffer.append(word)
                if last_NE:
                    # handle the last NE
                    ret.append(
                        (" ".join(ne_buffer), ne_start_pos, ne_last_pos, ne_type)
                    )
                it += 1
        else:
            tokens = self.__tokenize(sentence)
            nes = self.__ner(tokens)

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
                        ret.append(
                            (" ".join(ne_buffer), ne_start_pos, ne_last_pos, ne_type)
                        )
                else:
                    if (not last_NE) or (ne_type != ne):
                        if last_NE:
                            # append last ne
                            ret.append(
                                (
                                    " ".join(ne_buffer),
                                    ne_start_pos,
                                    ne_last_pos,
                                    ne_type,
                                )
                            )
                        # new entity
                        ne_start_pos = it
                        ne_last_pos = it + len(word)
                        ne_type = ne
                        ne_buffer = [word]
                        last_NE = True
                    else:
                        ne_last_pos = it + len(word)
                        ne_buffer.append(word)
                it += len(word)
            if last_NE:
                # handle the last NE
                ret.append((" ".join(ne_buffer), ne_start_pos, ne_last_pos, ne_type))
        return ret

    def get_parser(self, sentence):
        """
        :Data Requirements: :py:data:`.TProcess.StanfordParser`
        :Package Requirements: * **Java**

        This method uses Stanford LexParser to generate a lexical tree.
        """
        if self.__parser is None:
            self.__parser = DataManager.load("TProcess.StanfordParser")
        return str(list(self.__parser(sentence))[0])

    def get_wsd(self, tokens_and_pos):
        """
        :Data Requirements: :py:data:`.TProcess.NLTKWordNet`

        This method uses NTLK WordNet to generate synsets, and uses "lesk" algorithm which
        is proposed by Michael E. Lesk in 1986, to screen the sense out.
        """
        if self.__wordnet is None:
            self.__wordnet = DataManager.load("TProcess.NLTKWordNet")

        def lesk(sentence, word, pos):
            sent = set(sentence)
            synsets = self.__wordnet.synsets(word)
            if pos is not None:
                synsets = [ss for ss in synsets if str(ss.pos()) == pos]
            if len(synsets) == 0:
                return word
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
            ret.append(lesk(sentoken, word, pos))
        return ret

    def detokenizer(self, tokens):
        """
        :param list tokens: A list of token.
        :return: A detokenized sentence.
        :rtype: str
        
        This method is the inverse function of get_tokens which reads a list of tokens and returns a sentence.
        """
        all_tuple = True
        for it in tokens:
            if not isinstance(it, tuple):
                all_tuple = False
        if all_tuple:
            tokens = list(map(lambda x:x[0], tokens))
        
        ret = ""
        new_sent = True
        for token in tokens:
            if token in ".?!":
                ret += token
                new_sent = True
            elif len(token) >= 2 and token[0] == "'" and token[1] != "'":
                ret += token
            elif len(token) >= 2 and token[:2] == "##":
                ret += token[2:]
            elif token == "n't":
                ret += token
            else:
                if new_sent:
                    ret += " " + token.capitalize()
                    new_sent = False
                else:
                    ret += " " + token
        return ret