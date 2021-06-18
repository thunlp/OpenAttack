
from typing import List, Tuple, Union


class Tokenizer:
    """
    Tokenizer is the base class of all tokenizers.
    """

    def tokenize(self, x : str, pos_tagging : bool = True) -> Union[ List[str], List[Tuple[str, str]] ]:
        """
        Args:
            x: A sentence.
            pos_tagging: Whether to return Pos Tagging results.

        Returns:
            A list of tokens if **pos_tagging** is `False`
            
            A list of (token, pos) tuples if **pos_tagging** is `True`
        
        POS tag must be one of the following tags: ``["noun", "verb", "adj", "adv", "other"]``

        """
        return self.do_tokenize(x, pos_tagging)
    
    def detokenize(self, x : Union[List[str], List[Tuple[str, str]]]) -> str:
        """
        Args:
            x: The result of :py:meth:`.Tokenizer.tokenize`, can be a list of tokens or tokens with POS tags.
        Returns:
            A sentence.
        """
        if not isinstance(x, list):
            raise TypeError("`x` must be a list of tokens")
        if len(x) == 0:
            return ""
        x = [ it[0] if isinstance(it, tuple) else it for it in x ]
        return self.do_detokenize(x)

    
    def do_tokenize(self, x, pos_tagging):
        raise NotImplementedError()
    
    def do_detokenize(self, x):
        raise NotImplementedError()