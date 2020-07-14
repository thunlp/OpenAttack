import abc


class Attacker(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, clsf, input_, target=None):
        pass
