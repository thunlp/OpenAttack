from . import DefaultAttackerEval
from .. import Classifier
import json
from tqdm import tqdm

class InvokeLimitException(Exception):
    pass
class InvokeLimitWrapper(Classifier):
    def __init__(self, clsf, invoke_limit):
        self.invoke_limit = invoke_limit
        self.clsf = clsf
        self.clear()
    
    def clear(self):
        self.invoke = 0
    
    def get_invoke(self):
        return self.invoke
    
    def get_pred(self, input_):
        self.invoke += 1
        if self.invoke > self.invoke_limit:
            raise InvokeLimitException()
        return self.clsf.get_pred(input_)
    
    def get_prob(self, input_):
        self.invoke += 1
        if self.invoke > self.invoke_limit:
            raise InvokeLimitException()
        return self.clsf.get_prob(input_)
    
    def get_grad(self, input_, labels):
        self.invoke += 1
        if self.invoke > self.invoke_limit:
            raise InvokeLimitException()
        return self.clsf.get_grad(input_)

class InvokeLimitedAttackerEval(DefaultAttackerEval):
    def __init__(self, attacker, classifier, invoke_limit=100,
                    average_invoke=False, progress_bar=True, **kwargs):
        super().__init__(attacker, classifier, **kwargs)
        self.attacker = attacker
        self.classifier = InvokeLimitWrapper(classifier, invoke_limit)
        self.progress_bar = progress_bar

        self.average_invoke = average_invoke

    def update(self, sentA, sentB, out_of_invoke_limit):
        super().update(sentA, sentB)
        if self.average_invoke and sentB is not None:
            if "invoke" not in self.__result:
                self.__result["invoke"] = 0
            self.__result["invoke"] += self.classifier.get_invoke()
        if out_of_invoke_limit:
            if "out_of_invoke" not in self.__result:
                self.__result["out_of_invoke"] = 0
            self.__result["out_of_invoke"] += 1

    def eval_results(self, dataset):
        self.clear()
        for sent in dataset:
            self.classifier.clear()
            try:
                if isinstance(sent, tuple):
                    res = self.attacker(self.classifier, sent[0], sent[1])
                    if res is None:
                        yield (sent[0], None, None, self.update(sent[0], None, False) )
                    else:
                        yield (sent[0], res[0], res[1], self.update(sent[0], res[0], False))
                else:
                    res = self.attacker(self.classifier, sent)
                    if res is None:
                        yield (sent, None, None, self.update(sent, None, False) )
                    else:
                        yield (sent, res[0], res[1], self.update(sent, res[0], False))
            except InvokeLimitException:
                if isinstance(sent, tuple):
                    yield (sent[0], None, None, self.update(sent[0], None, True) )
                else:
                    yield (sent, None, None, self.update(sent, None, True) )
    
    def clear(self):
        super().clear()
        self.__result = {}

    
    def get_result(self):
        ret = super().get_result()
        if self.average_invoke and "invoke" in self.__result:
            ret["invoke"] = self.__result["invoke"] / ret["succeed"]
        if "out_of_invoke" in self.__result:
            ret["out_of_invoke"] = self.__result["out_of_invoke"]
        else:
            ret["out_of_invoke"] = 0
        return ret
