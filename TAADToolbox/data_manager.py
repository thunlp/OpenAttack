import pickle
import os
import urllib
from .exceptions import UnknownDataException, DataNotExistException
from .data import data_list


class DataManager(object):

    AVAILABLE_DATAS = [x["name"] for x in data_list]

    data_path = {
        x["name"]: os.path.join(os.getcwd(), "data", x["name"]) for x in data_list
    }

    data_download = {x["name"]: x["download"] for x in data_list}

    data_loader = {x["name"]: x["load"] for x in data_list}

    data_reference = {kw: None for kw in AVAILABLE_DATAS}

    __auto_download = True

    def __init__(self):
        raise NotImplementedError

    @classmethod
    def load(cls, data_name, cached=True):
        """
        Write usage here!
        """
        if data_name not in cls.AVAILABLE_DATAS:
            raise UnknownDataException

        if not os.path.exists(cls.data_path[data_name]):
            if cls.__auto_download:
                cls.download(data_name)
            else:
                raise DataNotExistException(data_name, cls.data_path[data_name])

        if not cached:
            return cls.data_loader[data_name](cls.data_path[data_name])
        elif cls.data_reference[data_name] is None:
            try:
                cls.data_reference[data_name] = cls.data_loader[data_name](
                    cls.data_path[data_name]
                )
            except OSError:
                raise DataNotExistException(data_name, cls.data_path[data_name])
        return cls.data_reference[data_name]
    
    @classmethod
    def setAutoDownload(cls, enabled=True):
        cls.__auto_download = enabled

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
    def download(cls, data_name, path=None, force=False):
        if data_name not in cls.AVAILABLE_DATAS:
            raise UnknownDataException
        if path is None:
            path = cls.data_path[data_name]
        if os.path.exists(path) and not force:
            return True
        cls.data_download[data_name](path)
        return True
