from . import AttackEvalBase
import json, sys, time
from tqdm import tqdm
from ..utils import visualizer, result_visualizer
from ..exceptions import ClassifierNotSupportException

class DefaultAttackEval(AttackEvalBase):
    """
    This class is a default implementation of AttackEval.
    """
    def __init__(self, attacker, classifier, running_time=True, progress_bar=True, **kwargs):
        """
        :param Attacker attacker: The attacker you use.
        :param Classifier classifier: The classifier you want to attack.
        :param bool running_time: If true, returns "Avg. Running Time" in summary. **Default:** True
        :param bool progress_bar: Dispaly a progress bar(tqdm). **Default:** True
        :param kwargs: Other parameters, see :py:class:`.AttackEvalBase` for detail.
        """
        super().__init__(**kwargs)
        self.attacker = attacker
        self.classifier = classifier
        self.__progress_bar = progress_bar
        self.__running_time = running_time
    
    def eval(self, dataset, total_len=None, visualize=False):
        """
        :param dataset: A list of data or a generator of data, each element can be a single sentence or a tuple of (sentence, label). A single sentence means an untargeted attack while tuple means a label-targeted attack.
        :type dataset: list or generator
        :param int total_len: If `dataset` is a generator, total_len is passed the progress bar.
        :param bool visualize: Display a visualized result for each instance and the summary.
        :return: Returns a dict of the summary.
        :rtype: dict

        In this method, ``eval_results`` is called and gets the result for each instance iteratively.
        """
        self.clear()
        if isinstance(dataset, list):
            total_len = len(dataset)
        
        counter = 0

        def tqdm_writer(x):
            return tqdm.write(x, end="")

        time_start = time.time()
        for x_orig, x_adv, y_adv, info in (tqdm(self.eval_results(dataset), total=total_len) if self.__progress_bar else self.eval_results(dataset)):
            counter += 1
            if visualize:
                try:
                    if x_adv is not None:
                        res = self.classifier.get_prob([x_orig, x_adv])
                        y_orig = res[0]
                        y_adv = res[1]
                    else:
                        y_orig = self.classifier.get_prob([x_orig])[0]
                except ClassifierNotSupportException:
                    if x_adv is not None:
                        res = self.classifier.get_pred([x_orig, x_adv])
                        y_orig = int(res[0])
                        y_adv = int(res[1])
                    else:
                        y_orig = int(self.classifier.get_pred([x_orig])[0])

                if self.__progress_bar:
                    visualizer(counter, x_orig, y_orig, x_adv, y_adv, info, tqdm_writer)
                else:
                    visualizer(counter, x_orig, y_orig, x_adv, y_adv, info, sys.stdout.write)
        
        res = self.get_result()
        if self.__running_time:
            res["Avg. Running Time"] = (time.time() - time_start) / counter

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
        """
        :param dataset: A list of data or a generator of data, each element can be a single sentence or a tuple of (sentence, label). A single sentence means an untargeted attack while tuple means a label-targeted attack.
        :type dataset: list or generator
        :return: A generator which generates the result for each instance, *(x_orig, x_adv, y_adv, info)*.
        :rtype: generator
        """
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