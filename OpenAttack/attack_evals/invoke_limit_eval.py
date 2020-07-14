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

        if self.average_invoke and info["Succeed"]:
            info["Queries"] = self.classifier.get_invoke()

        if out_of_invoke_limit:
            info["Query Exceeded"] = True
        else:
            info["Query Exceeded"] = False
        
        return self.update(info)
    
    def update(self, info):
        super().update(info)
        if "Queries" in info:
            if "invoke" not in self.__result:
                self.__result["invoke"] = 0
            self.__result["invoke"] += info["Queries"]
        
        if info["Query Exceeded"]:
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
                        info = self.__update(sent[0], None, False)
                        self.classifier.clear()
                        yield (sent[0], None, None, info )
                    else:
                        info = self.__update(sent[0], res[0], False)
                        self.classifier.clear()
                        yield (sent[0], res[0], res[1], info)
                else:
                    res = self.attacker(self.classifier, sent)
                    if res is None:
                        info = self.__update(sent, None, False)
                        self.classifier.clear()
                        yield (sent, None, None, info )
                    else:
                        info = self.__update(sent, res[0], False)
                        self.classifier.clear()
                        yield (sent, res[0], res[1], info)
            except InvokeLimitException:
                if isinstance(sent, tuple):
                    info = self.__update(sent[0], None, True)
                    self.classifier.clear()
                    yield (sent[0], None, None, info )
                else:
                    info = self.__update(sent, None, True)
                    self.classifier.clear()
                    yield (sent, None, None, info )
    
    def clear(self):
        super().clear()
        self.__result = {}

    
    def get_result(self):
        ret = super().get_result()
        if self.average_invoke and "invoke" in self.__result:
            ret["Avg. Victim Model Queries"] = self.__result["invoke"] / ret["Successful Instances"]
        return ret
