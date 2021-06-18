class Tag(object):
    def __init__(self, tag_name : str, type_ = None):
        self.__tag_name = tag_name
        self.__type : str = type_ if type_ is not None else ""
    
    @property
    def type(self) -> str:
        return self.__type
    
    @property
    def name(self) -> str:
        return self.__tag_name
    
    def __str__(self) -> str:
        return self.type + ":" + self.__tag_name
    
    def __eq__(self, o: object):
        return str(o).lower() == str(self).lower()
    
    def __hash__(self) -> int:
        return hash(str(self))
    
    def __repr__(self) -> str:
        return "<%s>" % str(self)
    