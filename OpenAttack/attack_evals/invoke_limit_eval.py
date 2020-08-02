from .default import DefaultAttackEval
from ..classifier import Classifier
from ..attacker import Attacker
import json
from tqdm import tqdm

class InvokeLimitException(Exception):
    pass

class InvokeLimitClassifierWrapper(Classifier):
    def __init__(self, clsf, invoke_limit):
        self.__invoke_limit = invoke_limit
        self.__clsf = clsf
        self.__brk = False
        self.__invoke = 0

    def clear(self):
        self.__invoke = 0
    
    def test(self, limit=True):
        self.__brk = limit
    
    def get_invoke(self):
        return self.__invoke
    
    def get_pred(self, input_, meta):
        if self.__brk and self.__invoke >= self.__invoke_limit:
            raise InvokeLimitException()
        self.__invoke += len(input_)
        return self.__clsf.get_pred(input_, meta)
    
    def get_prob(self, input_, meta):
        if self.__brk and self.__invoke >= self.__invoke_limit:
            raise InvokeLimitException()
        self.__invoke += len(input_)
        return self.__clsf.get_prob(input_, meta)
    
    def get_grad(self, input_, labels, meta):
        if self.__brk and self.__invoke > self.__invoke_limit:
            raise InvokeLimitException()
        self.__invoke += len(input_)
        return self.__clsf.get_grad(input_, labels, meta)

class InvokeLimitAttackerWrapper(Attacker):
    def __init__(self, attacker, clsf):
        self.__attacker = attacker
        self.__clsf = clsf
        self.__exceed = False
    
    def __call__(self, *args, **kwargs):
        self.__clsf.test()
        self.__clsf.clear()
        self.__exceed = False
        try:
            ret = self.__attacker(*args, **kwargs)
        except InvokeLimitException:
            ret = None
            self.__exceed = True
        self.__clsf.test(limit=False)
        return ret
    
    def exceed(self):
        return self.__exceed

class InvokeLimitedAttackEval(DefaultAttackEval):
    """
    Evaluate attackers and classifiers with invoke limitation.
    """
    def __init__(self, attacker, classifier, invoke_limit=100,
                    average_invoke=False, **kwargs):
        """
        :param Attacker attacker: The attacker you use.
        :param Classifier classifier: The classifier you want to attack.
        :param int invoke_limit: Limitation of invoke for each instance.
        :param bool average_invoke: If true, returns "Avg. Victim Model Queries".
        :param kwargs: Other parameters, see :py:class:`.DefaultAttackEval` for detail.
        """
        super().__init__(attacker, classifier, **kwargs)

        # wrap classifier, attacker after super().__init__
        self.classifier = InvokeLimitClassifierWrapper(self.classifier, invoke_limit)
        self.attacker =  InvokeLimitAttackerWrapper(self.attacker, self.classifier)

        # keep a private version
        self.__attacker = self.attacker
        self.__classifier = self.classifier

        self.__average_invoke = average_invoke
    
    def measure(self, sentA, sentB):
        info = super().measure(sentA, sentB)
        if self.__attacker.exceed():
            info["Query Exceeded"] = True
        else:
            info["Query Exceeded"] = False

        # only records succeed attacks
        if info["Succeed"] and self.__average_invoke:
            info["Queries"] = self.__classifier.get_invoke()
        return info

    def update(self, info):
        info = super().update(info)
        if "Queries" in info:
            if "invoke" not in self.__result:
                self.__result["invoke"] = 0
            self.__result["invoke"] += info["Queries"]
        
        if info["Query Exceeded"]:
            if "out_of_invoke" not in self.__result:
                self.__result["out_of_invoke"] = 0
            self.__result["out_of_invoke"] += 1
        return info
    
    def clear(self):
        super().clear()
        self.__result = {}

    def get_result(self):
        ret = super().get_result()
        if self.__average_invoke and "invoke" in self.__result:
            ret["Avg. Victim Model Queries"] = self.__result["invoke"] / ret["Successful Instances"]
        return ret
