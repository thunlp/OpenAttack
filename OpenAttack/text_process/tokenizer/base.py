
from typing import List, Tuple, Union


class Tokenizer:
    def tokenize(self, x : str, pos_tagging=True) -> Union[ List[str], List[Tuple[str, str]] ]:
        return self.do_tokenize(x, pos_tagging)
    
    def detokenize(self, x : Union[List[str], List[Tuple[str, str]]]) -> str:
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