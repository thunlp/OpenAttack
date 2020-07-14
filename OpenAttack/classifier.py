import abc
from .exceptions import ClassifierNotSupportException


class Classifier(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_):
        return self.get_grad(input_, [0] * len(input_))[0]

    def get_grad(self, input_, labels):
        raise ClassifierNotSupportException()
