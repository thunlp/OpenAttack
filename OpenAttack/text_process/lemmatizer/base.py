class Lemmatizer:
    """
    Base class of all lemmatizers.
    """

    def lemmatize(self, token : str, pos : str) -> str:
        """
        Args:
            token: A token.
            pos: POS tag of input token.
        Returns:
            Lemma of this token.
        """
        return self.do_lemmatize(token, pos)
    
    def delemmatize(self, lemma : str, pos : str) -> str:
        """
        Args:
            lemma: A lemma of some token.
            pos: POS tag of input lemma.
        Returns:
            The original token.
        """
        return self.do_delemmatize(lemma, pos)
    
    def do_lemmatize(self, token, pos):
        raise NotImplementedError()
    
    def do_delemmatize(self, lemma, pos):
        raise NotImplementedError()
    