from . import AttackerEvalBase
import json

class DefaultAttackerEval(AttackerEvalBase):
    def __init__(self, attacker, classifier, **kwargs):
        super().__init__(**kwargs)
        self.attacker = attacker
        self.classifier = classifier
    
    def eval(self, dataset):
        self.clear()
        for sent in dataset:
            if isinstance(sent, tuple):
                res = self.attacker(self.classifier, sent[0], sent[1])
                if res is None:
                    self.update(sent[0], None)
                else:
                    self.update(sent[0], res[0])
            else:
                res = self.attacker(self.classifier, sent)
                if res is None:
                    self.update(sent, None)
                else:
                    self.update(sent, res[0])
        return self.get_result()

    def print(self):
        print( json.dumps( self.get_result(), indent="\t" ) )

    def dump(self, file_like_object):
        json.dump( self.get_result(), file_like_object )

    def dumps(self):
        return json.dumps()

    def eval_results(self, dataset):
        self.clear()
        for sent in dataset:
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