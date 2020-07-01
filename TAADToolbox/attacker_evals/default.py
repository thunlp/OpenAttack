from . import AttackerEvalBase

class DefaultAttackerEval(AttackerEvalBase):
    def __init__(self, attacker, classifier, **kwargs):
        super().__init__(**kwargs)
        self.attacker = attacker
        self.classifier = classifier
    
    def eval(self, dataset):
        pass

    def print(self):
        pass

    def dump(self, file_like_object):
        pass

    def dumps(self):
        pass

    def eval_results(self, dataset):
        pass