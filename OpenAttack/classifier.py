import abc
import functools
from .exceptions import ClassifierNotSupportException

class Classifier(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    def get_pred(self, input_, meta):
        """
        :param list input_: A list of sentences which we want to predict.
        :param dict meta: Meta of DataInstance (**Optional parameter**).
        :return: The result is a 1-dim numpy.array, *[pred1, pred2, ...]*.
        :rtype: numpy.array(dtype=numpy.long)

        This is used to get the predictions of a batch of sentences.
        """
        return self.get_prob(input_, meta).argmax(axis=1)

    def get_prob(self, input_, meta):
        """
        :param list input_: A list of sentences of which we want to get the probabilities.
        :param dict meta: Meta of DataInstance (**Optional parameter**).
        :return: The result is a 2-dim numpy.array. The first dimension represents sentences and the second dimension represents probabilities of every type.
        :rtype: numpy.array(dtype=numpy.float)

        This is used to get the probabilities of a batch of sentences on every type.
        """
        from .text_processors import DefaultTextProcessor
        processor = DefaultTextProcessor()
        x_batch = [ list(map(lambda x:x[0], processor.get_tokens(sent))) for sent in input_ ]
        return self.get_grad(x_batch, [0] * len(x_batch), meta)[0]

    def get_grad(self, input_, labels, meta):
        """
        :param list input_: A list of lists of tokens of which we want to get the gradients.
        :param list labels: A list of types based on which we want to get the gradients.
        :param dict meta: Meta of DataInstance (**Optional parameter**).
        :return: The result is a tuple of 2 numpy.array. The first numpy.array is same as the return of get_prob and the second numpy.array is gradients on the model's input, not the input\_. So the shape of the second numpy.array is the same model input.
        :rtype: tuple

        This is used to get the gradients of a batch of sentences(presented as lists of tokens) on the appointed types.
        """
        raise ClassifierNotSupportException()


    def __hook(self, func, name, *args, **kwargs):
        fuclen = func.__code__.co_argcount - 1
        reqlen = 2 if name == "get_grad" else 1
        if len(args) > reqlen + 1:
            raise TypeError("Classifier.%s get too many arguments" % name)
        if len(args) < reqlen:
            raise TypeError("Classifier.%s missing %d required positional argument" % (name, reqlen - len(args)))
        if not isinstance(args[0], list):
            raise TypeError("input_ should be a list of %s" % ("tokens" if name == "get_grad" else "str"))
        if fuclen == reqlen:
            args = args[:reqlen]
        elif len(args) < fuclen:
            args = tuple([*args] + [{}])

        return func(*args, **kwargs)

    
    def __getattribute__(self, name):
        ret = super().__getattribute__(name)
        if name in ["get_pred", "get_prob", "get_grad"]:
            return functools.partial(self.__hook, ret, name)
        return ret

