import abc


class Attacker(metaclass=abc.ABCMeta):
    """
    This is the base class for attackers.
    """
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, clsf, input_, target=None):
        """
        :param Classifier clsf: The classifier to be attacked.
        :param str input_: The original sentence.
        :param target: If it's `None`, plays an untargeted attack; If it's label, plays a targeted attack.
        :type target: int or None
        :return: Attack result (adversarial_sentence, adversarial_label), or None if failed.
        :rtype: tuple or None
        """
        pass
