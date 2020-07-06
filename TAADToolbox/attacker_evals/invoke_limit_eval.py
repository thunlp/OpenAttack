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
    def __init__(self, attacker, classifier, invoke_limit=100, progress_bar=True, **kwargs):
        super().__init__(attacker, classifier, **kwargs)
        self.attacker = attacker
        self.classifier = InvokeLimitWrapper(classifier, invoke_limit)
        self.progress_bar = progress_bar
    
    def eval(self, dataset):
        self.clear()
        for _ in (tqdm(self.eval_results(dataset)) if self.progress_bar else self.eval_results(dataset)):
            pass
        return self.get_result()

    def eval_results(self, dataset):
        self.clear()
        for sent in dataset:
            self.classifier.clear()
            try:
                if isinstance(sent, tuple):
                    res = self.attacker(self.classifier, sent[0], sent[1])
                    if res is None:
                        yield (sent[0], None, None, self.update(sent[0], None) )
                    else:
                        yield (sent[0], res[0], res[1], self.update(sent[0], res[0]))
                else:
                    res = self.attacker(self.classifier, sent)
                    if res is None:
                        yield (sent, None, None, self.update(sent, None) )
                    else:
                        yield (sent, res[0], res[1], self.update(sent, res[0]))
            except InvokeLimitException:
                if isinstance(sent, tuple):
                    yield (sent[0], None, None, self.update(sent[0], None) )
                else:
                    yield (sent, None, None, self.update(sent, None) )
