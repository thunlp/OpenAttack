import pkgutil
import pickle
import urllib
from tqdm import tqdm
from ..exceptions import DataConfigErrorException


def load_data():
    def pickle_loader(path):
        return pickle.load(open(path, "rb"))

    def url_downloader(url, resource_name):
        if url[0] == "/":
            remote_url = url[1:]
        else:
            remote_url = url

        def DOWNLOAD(path : str, source : str):
            CHUNK_SIZE = 1024
            if not source.endswith("/"):
                source = source + "/"
            with urllib.request.urlopen(source + remote_url) as fin:
                total_length = int(fin.headers["content-length"])
                with open(path, "wb") as fout:
                    with tqdm(total=total_length, unit="B", desc="Downloading %s" % resource_name, unit_scale=True) as pbar:                            
                        while True:
                            data = fin.read(CHUNK_SIZE)
                            if len(data) == 0:
                                break
                            fout.write(data)
                            pbar.update( len(data) )
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
                tmp["download"] = url_downloader(data.DOWNLOAD, data.NAME)
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
