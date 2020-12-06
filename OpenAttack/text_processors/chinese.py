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
        :Data Requirements: :py:data:`TProcess.NLTKSentTokenizer` , :py:data:`.TProcess.NLTKPerceptronPosTagger`

        This method uses ``nltk.WordPunctTokenizer()`` for word tokenization and
        ``nltk.tag.PerceptronTagger()`` to generate POS tag in "Penn tagset".
        """
        import thulac
        thu = thulac.thulac()
        lists = thu.cut(sentence)
        ans = []
        for i in range(len(lists)):
            ans.append(tuple(lists[i]))
        return ans
