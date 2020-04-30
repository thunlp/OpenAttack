import pickle
import os
import urllib
from .exceptions import UnknownDataException, DataNotExistException


class DataManager(object):

    AVAILABLE_DATAS = ["example"]

    data_path = {"example": "/example"}

    data_download = {"example": "https://data.to.example/example"}

    data_reference = {kw: None for kw in AVAILABLE_DATAS}

    def __init__(self):
        raise NotImplementedError

    @classmethod
    def load(cls, data_name):
        """
        Write usage here!
        """
        if data_name not in cls.AVAILABLE_DATAS:
            raise UnknownDataException
        if cls.data_reference[data_name] is None:
            try:
                cls.data_reference[data_name] = pickle.load(
                    open(cls.get(data_name), "rb")
                )
            except OSError:
                raise DataNotExistException
        return cls.data_reference[data_name]

    @classmethod
    def get(cls, data_name):
        if data_name not in cls.AVAILABLE_DATAS:
            raise UnknownDataException
        return cls.data_path[data_name]

    @classmethod
    def set_path(cls, path, data_name=None):
        if data_name is None:
            nw_dict = {}
            for kw, pt in cls.data_path.items():
                nw_dict[kw] = os.path.join(path, os.path.basename(pt))
            cls.data_path = nw_dict
        else:
            if data_name not in cls.AVAILABLE_DATAS:
                raise UnknownDataException
            cls.data_path[data_name] = path

    @classmethod
    def download(cls, data_name, path=None):
        if data_name not in cls.AVAILABLE_DATAS:
            raise UnknownDataException
        if path is None:
            path = cls.data_path[data_name]
        with urllib.request.urlopen(cls.data_download[data_name]) as f:
            open(path, "wb").write(f.read())
        return True
