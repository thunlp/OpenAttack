def NLIWrapper(func):
    """
    ``NLIWrapper`` is a "decorator" used in NLI classifiers to provide a more friendly interface.

    .. code-block:: python
        :linenos:

        # original interface
        def get_pred(self, input_, meta):
            pass

        # new interface
        @NLIWrapper
        def get_pred(self, hypothesis, reference):
            pass
    
    ``hypothesis`` and ``reference`` are both lists of sentences with the same lengths.
    """
    func_name = func.__code__.co_name
    if func_name in ["get_pred", "get_prob"]:
        def warpper1(self, input_, meta):
            refs = [ meta["reference"] ] * len(input_)
            return func(self, input_, refs)
        return warpper1
    elif func_name in ["get_grad"]:
        def wrapper2(self, input_, labels, meta):
            refs = [ meta["reference"] ] * len(input_)
            return func(self, input_, refs, labels)
        return wrapper2
    else:
        return func