import abc


class AttackerEval(metaclass=abc.ABCMeta):
    def __init__(self, attacker, classifier, **kwargs):
        pass

    @abc.abstractmethod
    def eval(self, dataset):
        pass

    @abc.abstractmethod
    def print(self):
        pass

    @abc.abstractmethod
    def dump(self, file_like_object):
        pass

    @abc.abstractmethod
    def dumps(self):
        pass

    @abc.abstractmethod
    def eval_results(self, dataset):
        pass
