from ..method import VictimMethod
from ...attack_assist.word_embedding import WordEmbedding

class GetPredict(VictimMethod):
    def before_call(self, input_):
        if not isinstance(input_, list):
            raise TypeError( "get_pred: `input` must be a list of sentences, but got %s" % type(input_) )
        if len(input_) == 0:
            raise ValueError("empty `input` list")
        for i, it in enumerate(input_):
            if not isinstance(it, str):
                raise  TypeError( "get_pred: `input[%d]` must be a list of sentences, but got %s" % (i, type(it)) )
    
    def invoke_count(self, input_):
        return len(input_)

class GetProbability(VictimMethod):
    def before_call(self, input_):
        if not isinstance(input_, list):
            raise TypeError( "get_prob: `input` must be a list of sentences, but got %s" % type(input_) )
        if len(input_) == 0:
            raise ValueError("empty `input` list")
        for i, it in enumerate(input_):
            if not isinstance(it, str):
                raise  TypeError( "get_prob: `input[%d]` must be a sentence, but got %s" % (i, type(it)) )

    def invoke_count(self, input_):
        return len(input_)

class GetGradient(VictimMethod):
    def before_call(self, input_, labels):
        if not isinstance(input_, list):
            raise TypeError( "get_grad: `input` must be a list of token lists, but got %s" % type(input_) )
        if len(input_) == 0:
            raise ValueError("empty `input` list")
        for i, it in enumerate(input_):
            if not isinstance(it, list):
                raise  TypeError( "get_grad: `input[%d]` must be a token list, but got %s" % (i, type(it)) )
            for j, token in enumerate(it):
                if not isinstance(token, str):
                    raise  TypeError( "get_grad: `input[%d][%d]` must be a token, but got %s" % (i, j, type(it)) )

        if len(input_) != len(labels):
            raise ValueError("`input_` and `labels` must be the same length. (%d != %d)" % (len(input_), len(labels)))

    def invoke_count(self, input_, labels):
        return len(input_)

class GetEmbedding(VictimMethod):
    def after_call(self, ret):
        if not isinstance(ret, WordEmbedding):
            raise TypeError("`get_embedding`: must return a `WordEmbedding` object")