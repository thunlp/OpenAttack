from .base import Tokenizer
import transformers
class TransformersTokenizer(Tokenizer):
    """
    Pretrained Tokenizer from transformers.

    Usually returned by :py:class:`.TransformersClassifier` .
    
    """

    @property
    def TAGS(self):
        return { self.__lang_tag }

    def __init__(self, tokenizer : transformers.PreTrainedTokenizerBase, lang_tag):
        self.__tokenizer = tokenizer
        self.__lang_tag = lang_tag

    def do_tokenize(self, x, pos_tagging):
        if pos_tagging:
            raise ValueError("`%s` does not support pos tagging" % self.__class__.__name__)
        return self.__tokenizer.tokenize(x)
    
    def do_detokenize(self, x):
        return self.__tokenizer.convert_tokens_to_string(x)
        