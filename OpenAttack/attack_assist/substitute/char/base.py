from typing import List, Tuple

class CharSubstitute(object):
    def __call__(self, char : str):
        """
        :param char: the raw char
        :return: The result is a list of tuples, *(substitute, distance)*.
        :rtype: list of tuple

        In CharSubstitute, we return a list of chars that are visually similar to the original word.
        """
        return self.substitute(char)
    
    def substitute(self, char : str) -> List[Tuple[str, float]]:
        raise NotImplementedError()
        