import os
from typing import Any, Optional
from .exceptions import UnknownDataException, DataNotExistException
from .data import data_list
import inspect


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

    source = "https://data.thunlp.org/"

    def __init__(self):
        raise NotImplementedError()
    
    @classmethod
    def enable_cdn(cls):
        """
        Enable cdn for all official downloads.
        """

        cls.source = "https://cdn.data.thunlp.org/"
    
    @classmethod
    def disable_cdn(cls):
        """
        Disable cdn for all official downloads.
        """
        cls.source = "https://data.thunlp.org/"

    @classmethod
    def load(cls, data_name : str, cached : bool = True) -> Any:
        """
        Load data from local storage, and download it automatically if not exists.

        Args:
            data_name: The name of resource that you want to load. You can find all the available resource names in ``DataManager.AVAILABLE_DATAS``. *Note: all the names are* **CASE-SENSITIVE**.
            cached: If **cached** is *True*, DataManager will lookup the cache before load it to avoid duplicate disk IO. If **cached** is *False*, DataManager will directly load data from disk. **Default:** *True*.
        
        Returns:
            The object returned by LOAD function of corresponding data.

        Raises:
            UnknownDataException: For loading an unavailable data. 
            DataNotExistException:  For loading a data that has not been downloaded. This appends when AutoDownload mechanism is disabled.

        """

        if data_name not in cls.AVAILABLE_DATAS:
            raise UnknownDataException()

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
    def setAutoDownload(cls, enabled : bool = True):
        """
        AutoDownload mechanism is enabled by default.

        Args:
            enabled: Change if DataManager automatically download the data when loading.
        
        """
        cls.__auto_download = enabled

    @classmethod
    def get(cls, data_name : str) -> str:
        """
        Args:
            data_name: The name of data.
        Returns:
            Relative path of data.

        """
        if data_name not in cls.AVAILABLE_DATAS:
            raise UnknownDataException
        return cls.data_path[data_name]

    @classmethod
    def set_path(cls, path : str, data_name : Optional[str] = None):
        """Set the path for a specific data or for all data.

        If **data_name** is *None*, all paths will be changed to corresponding file under **path** directory.

        If **data_name** is *not None*, the specific data path will be changed to **path**.

        The default paths for all data are ``./data/<data_name>``, and you can manually change them using this method .

        Args:
            path: The path to data, or path to the directory where all data is stored.
            data_name: The name of data. If **data_name** is *None*, all paths will be changed.


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
    def download(cls, data_name : str, path : Optional[str] = None, force : bool = False):
        """
        This method will check if data exists before getting it from "Data Server".You can use **force** to skip this step.

        Args:
            data_name: Name of the data that you want to download.
            path: Specify a path when before download. Leaves None for download to default **path**.
            force: Force download the data.

        Raises:
            UnknownDataException: For downloading an unavailable data.

        
        """
        if data_name not in cls.AVAILABLE_DATAS:
            raise UnknownDataException()
        if path is None:
            path = cls.data_path[data_name]
        if os.path.exists(path) and not force:
            return True
        download_func = cls.data_download[data_name]

        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        num_args = len(inspect.getfullargspec(download_func).args)
        if num_args == 1:
            download_func(path)
        elif num_args == 2:
            download_func(path, cls.source)
        return True
