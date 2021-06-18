from typing import List, Tuple

class CharSubstitute(object):
    def __call__(self, char : str)  -> List[Tuple[str, float]]:
        """Char-level substitute algorithm.

        In CharSubstitute, we return a list of chars that are visually similar to the original word.

        Args:
            char: A signle char
        
        Returns:
            A list of chars and distance to original char (distance is a number between 0 and 1, with smaller indicating more similarity).
        
        """
        return self.substitute(char)
    
    def substitute(self, char : str) -> List[Tuple[str, float]]:
        raise NotImplementedError()
        