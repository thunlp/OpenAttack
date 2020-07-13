from . import DefaultAttackEval
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
        if self.invoke >= self.invoke_limit:
            raise InvokeLimitException()
        self.invoke += len(input_)
        return self.clsf.get_pred(input_)
    
    def get_prob(self, input_):
        if self.invoke >= self.invoke_limit:
            raise InvokeLimitException()
        self.invoke += len(input_)
        return self.clsf.get_prob(input_)
    
    def get_grad(self, input_, labels):
        if self.invoke > self.invoke_limit:
            raise InvokeLimitException()
        self.invoke += len(input_)
        return self.clsf.get_grad(input_, labels)

class InvokeLimitedAttackEval(DefaultAttackEval):
    def __init__(self, attacker, classifier, invoke_limit=100,
                    average_invoke=False, progress_bar=True, **kwargs):
        super().__init__(attacker, classifier, **kwargs)
        self.attacker = attacker
        self.classifier = InvokeLimitWrapper(classifier, invoke_limit)
        self.progress_bar = progress_bar

        self.average_invoke = average_invoke

    def __update(self, sentA, sentB, out_of_invoke_limit):
        info = super().measure(sentA, sentB)

        if self.average_invoke and info["succeed"]:
            info["invoke"] = self.classifier.get_invoke()

        if out_of_invoke_limit:
            info["out_of_invoke"] = True
        else:
            info["out_of_invoke"] = False
        
        return self.update(info)
    
    def update(self, info):
        super().update(info)
        if "invoke" in info:
            if "invoke" not in self.__result:
                self.__result["invoke"] = 0
            self.__result["invoke"] += info["invoke"]
        
        if info["out_of_invoke"]:
            if "out_of_invoke" not in self.__result:
                self.__result["out_of_invoke"] = 0
            self.__result["out_of_invoke"] += 1
        return info

    def eval_results(self, dataset):
        self.clear()
        for sent in dataset:
            self.classifier.clear()
            try:
                if isinstance(sent, tuple):
                    res = self.attacker(self.classifier, sent[0], sent[1])
                    if res is None:
                        yield (sent[0], None, None, self.__update(sent[0], None, False) )
                    else:
                        yield (sent[0], res[0], res[1], self.__update(sent[0], res[0], False))
                else:
                    res = self.attacker(self.classifier, sent)
                    if res is None:
                        yield (sent, None, None, self.__update(sent, None, False) )
                    else:
                        yield (sent, res[0], res[1], self.__update(sent, res[0], False))
            except InvokeLimitException:
                if isinstance(sent, tuple):
                    yield (sent[0], None, None, self.__update(sent[0], None, True) )
                else:
                    yield (sent, None, None, self.__update(sent, None, True) )
    
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
