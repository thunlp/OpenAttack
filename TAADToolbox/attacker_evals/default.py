from . import AttackerEvalBase
import json
from tqdm import tqdm

class DefaultAttackerEval(AttackerEvalBase):
    def __init__(self, attacker, classifier, progress_bar=True, **kwargs):
        super().__init__(**kwargs)
        self.attacker = attacker
        self.classifier = classifier
        self.progress_bar = progress_bar
    
    def eval(self, dataset):
        self.clear()
        total_len = None
        if isinstance(dataset, list):
            total_len = len(dataset)
        for _ in (tqdm(self.eval_results(dataset), total=total_len) if self.progress_bar else self.eval_results(dataset)):
            pass
        return self.get_result()

    def print(self):
        print( json.dumps( self.get_result(), indent="\t" ) )

    def dump(self, file_like_object):
        json.dump( self.get_result(), file_like_object )

    def dumps(self):
        return json.dumps( self.get_result() )
    
    def __update(self, sentA, sentB):
        info = self.measure(sentA, sentB)
        return self.update(info)

    def eval_results(self, dataset):
        self.clear()
        for sent in dataset:
            if isinstance(sent, tuple):
                res = self.attacker(self.classifier, sent[0], sent[1])
                if res is None:
                    yield (sent[0], None, None, self.__update(sent[0], None) )
                else:
                    yield (sent[0], res[0], res[1], self.__update(sent[0], res[0]))
            else:
                res = self.attacker(self.classifier, sent)
                if res is None:
                    yield (sent, None, None, self.__update(sent, None) )
                else:
                    yield (sent, res[0], res[1], self.__update(sent, res[0]))