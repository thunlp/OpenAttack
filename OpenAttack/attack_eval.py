import abc


class AttackEval(metaclass=abc.ABCMeta):
    """
    This module is used to evaluate the performance of attackers and classifiers.
    """
    def __init__(self, attacker, classifier, **kwargs):
        """
        :param Attacker attacker: The attacker you use.
        :param Classifier classifier: The classifier you want to attack.

        Initialize AttackEval.
        """

    @abc.abstractmethod
    def eval(self, dataset):
        """
        :param list dataset: A list of data or a generator of data, each element can be a single sentence or a tuple of (sentence, label). A single sentence means an untargeted attack while tuple means a label-targeted attack.
        :return: A dict contains the results.
        :rtype: dict
        """
        pass

    @abc.abstractmethod
    def eval_results(self, dataset):
        """
        :param list dataset: A list of data or a generator of data, each element can be a single sentence or a tuple of (sentence, label). A single sentence means an untargeted attack while tuple means a label-targeted attack.
        :return: A `generator` generates result for every instance in dataset.
        :rtype: generator
        """
        pass
