class ConstituencyParser:
    """
    Base class of all constituency parsers.
    """

    def __call__(self, sentence : str) -> str:
        """
        Args:
            sentence: A sentecne.
        Returns:
            Constituency parser results.
        """
        return self.parse(sentence)
    
    def parse(self, sentence : str) -> str:
        raise NotImplementedError()