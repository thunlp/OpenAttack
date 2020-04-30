import abc


class Classifier(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_pred(self, input_):
        pass

    @abc.abstractmethod
    def get_prob(self, input_):
        pass

    @abc.abstractmethod
    def get_grad(self, input_, labels):
        pass
