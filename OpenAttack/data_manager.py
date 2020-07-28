import pickle
import os
import urllib
from .exceptions import UnknownDataException, DataNotExistException
from .data import data_list


class DataManager(object):
    """
    DataManager is a module that manages all the resources used in Attacker, Metric, Substitute, TextProcessors and utils.

    It reads configuration files in OpenAttack/data/\*.py, and initialize these resources when you load them.

    You can use 

    .. code-block:: python
    
        for data_name in OpenAttack.DataManager.AVAILABLE_DATAS:
            OpenAttack.download(data_name)
     
    to download all the available resources, but this is not recommend because of the huge network cost.

    ``OpenAttack.load`` and ``OpenAttack.download`` is a alias of 
    ``OpenAttack.DataManager.load`` and ``OpenAttack.DataManager.download``, they are exactly equivalent.
    These two methods are useful for both developer and user, that's the reason we provide shortter name for them.
    """

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
        :param str data_name: The name of resource that you want to load. You can find all the available resource names in ``DataManager.AVAILABLE_DATAS``. *Note: all the names are* **CASE-SENSITIVE**.
        :param bool cached: If **cached** is *True*, DataManager will lookup the cache before load it to avoid duplicate disk IO. If **cached** is *False*, DataManager will directly load data from disk. **Default:** *True*.
        :return: data, for details see the documentation for "Data" (size, type, description etc).
        :rtype: Any

        :raises UnknownDataException: For loading an unavailable data.
        :raises DataNotExistException:  For loading a data that has not been downloaded. This appends when AutoDownload mechanism is disabled.


        
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
    def loadDataset(cls, data_name, cached=True):
        """
        This method is equivalent to ``DataManager.load("Dataset." + data_name)``.
        :rtype: Dataset
        """
        return cls.load("Dataset." + data_name, cached=cached)
    
    @classmethod
    def loadVictim(cls, data_name, cached=True):
        """
        This method is equivalent to ``DataManager.load("Victim." + data_name)``.
        """
        return cls.load("Victim." + data_name, cached=cached)
    
    @classmethod
    def loadTProcess(cls, data_name, cached=True):
        """
        This method is equivalent to ``DataManager.load("TProcess." + data_name)``.
        """
        return cls.load("TProcess." + data_name, cached=cached)
    
    @classmethod
    def loadAttackAssist(cls, data_name, cached=True):
        """
        This method is equivalent to ``DataManager.load("AttackAssist." + data_name)``.
        """
        return cls.load("AttackAssist." + data_name, cached=cached)
    
    @classmethod
    def setAutoDownload(cls, enabled=True):
        """
        :param bool enabled: Change if DataManager automatically download the data when loading.
        :return: None

        AutoDownload mechanism is enabled by default.
        """
        cls.__auto_download = enabled

    @classmethod
    def get(cls, data_name):
        """
        :param str data_name: The name of data.
        :return: Relative path of data.
        :rtype: str
        """
        if data_name not in cls.AVAILABLE_DATAS:
            raise UnknownDataException
        return cls.data_path[data_name]

    @classmethod
    def set_path(cls, path, data_name=None):
        """
        :param str path: The path to data, or path to the directory where all data is stored.
        :param data_name: The name of data. If **data_name** is *None*, all paths will be changed.
        :type data_name: str or None
        :return: None
        :raises UnknownDataException: For changing an unavailable data.

        Set the path for a specific data or for all data.

        If **data_name** is *None*, all paths will be changed to corresponding file under **path** directory.

        If **data_name** is *not None*, the specific data path will be changed to **path**.

        The default paths for all data are ``./data/<data_name>``, and you can manually change them using this method .
        """
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
        """
        :param str data_name: Name of the data that you want to download.
        :param str path: Specify a path when before download. Leaves None for download to default **path**.
        :param bool force: Force download the data.
        :return: This method always returns True
        :rtype: bool
        :raises UnknownDataException: For downloading an unavailable data.

        This method will check if data exists before getting it from "Data Server".You can 
        use **force** to skip this step.
        """
        if data_name not in cls.AVAILABLE_DATAS:
            raise UnknownDataException
        if path is None:
            path = cls.data_path[data_name]
        if os.path.exists(path) and not force:
            return True
        cls.data_download[data_name](path)
        return True
