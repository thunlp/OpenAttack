import pkgutil
import pickle
import urllib
from ..exceptions import DataConfigErrorException


def load_data():
    def pickle_loader(path):
        return pickle.load(open(path, "rb"))

    def url_downloader(url):
        def DOWNLOAD(path):
            with urllib.request.urlopen(url) as f:
                open(path, "wb").write(f.read())
            return True

        return DOWNLOAD

    ret = []
    for data in pkgutil.iter_modules(__path__):
        data = data.module_finder.find_loader(data.name)[0].load_module()
        if hasattr(data, "NAME") and hasattr(data, "DOWNLOAD"):  # is a data module
            tmp = {"name": data.NAME}
            if callable(data.DOWNLOAD):
                tmp["download"] = data.DOWNLOAD
            elif isinstance(data.DOWNLOAD, str):
                tmp["download"] = url_downloader(data.DOWNLOAD)
            else:
                raise DataConfigErrorException(
                    "Data Module: %s\n dir: %s\n type: %s"
                    % (data, dir(data), type(data.DOWNLOAD))
                )

            if hasattr(data, "LOAD"):
                tmp["load"] = data.LOAD
            else:
                tmp["load"] = pickle_loader
            ret.append(tmp)
        else:
            pass  # this is not a data module
    return ret


data_list = load_data()
del load_data
