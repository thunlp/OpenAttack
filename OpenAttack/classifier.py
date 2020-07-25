import abc
from .exceptions import ClassifierNotSupportException


class Classifier(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    def get_pred(self, input_):
        """
        :param list input_: A list of sentences which we want to predict.
        :return: The result is a 1-dim numpy.array, *[pred1, pred2, ...]*.
        :rtype: numpy.array(dtype=numpy.long)

        This is used to get the predictions of a batch of sentences.
        """
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_):
        """
        :param list input_: A list of sentences of which we want to get the probabilities.
        :return: The result is a 2-dim numpy.array. The first dimension represents sentences and the second dimension represents probabilities of every type.
        :rtype: numpy.array(dtype=numpy.float)

        This is used to get the probabilities of a batch of sentences on every type.
        """
        return self.get_grad(input_, [0] * len(input_))[0]

    def get_grad(self, input_, labels):
        """
        :param list input_: A list of lists of tokens of which we want to get the gradients.
        :param list labels: A list of types based on which we want to get the gradients.
        :return: The result is a tuple of 2 numpy.array. The first numpy.array is same as the return of get_prob and the second numpy.array is gradients on the model's input, not the input\_. So the shape of the second numpy.array is the same model input.
        :rtype: tuple

        This is used to get the gradients of a batch of sentences(presented as lists of tokens) on the appointed types.
        """
        raise ClassifierNotSupportException()
