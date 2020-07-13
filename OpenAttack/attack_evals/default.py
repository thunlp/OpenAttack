from . import AttackEvalBase
import json, sys
from tqdm import tqdm
from ..utils import visualizer, result_visualizer

class DefaultAttackEval(AttackEvalBase):
    def __init__(self, attacker, classifier, progress_bar=True, **kwargs):
        super().__init__(**kwargs)
        self.attacker = attacker
        self.classifier = classifier
        self.progress_bar = progress_bar
    
    def eval(self, dataset, total_len=None, visualize=False):
        self.clear()
        if isinstance(dataset, list):
            total_len = len(dataset)
        
        counter = 0

        def tqdm_writer(x):
            return tqdm.write(x, end="")

        for x_orig, x_adv, y_adv, info in (tqdm(self.eval_results(dataset), total=total_len) if self.progress_bar else self.eval_results(dataset)):
            if visualize:
                counter += 1
                y_orig = self.classifier.get_pred([x_orig])[0]
                if self.progress_bar:
                    visualizer(counter, x_orig, y_orig, x_adv, y_adv, info, tqdm_writer)
                else:
                    visualizer(counter, x_orig, y_orig, x_adv, y_adv, info, sys.stdout.write)
        res = self.get_result()
        if visualize:
            result_visualizer(res, sys.stdout.write)
        return res

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