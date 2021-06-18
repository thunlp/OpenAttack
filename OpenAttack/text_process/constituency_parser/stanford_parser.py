from .base import ConstituencyParser
from ...tags import *
from ...data_manager import DataManager

class StanfordParser(ConstituencyParser):
    """
    Constituency parser based on stanford parser.
    
    :Requirements:
        * java
    
    """

    TAGS = { TAG_English }

    def __init__(self):
        self.__parser = DataManager.load("TProcess.StanfordParser")
    
    def parse(self, sentence: str) -> str:
        return str(list(self.__parser(sentence))[0])