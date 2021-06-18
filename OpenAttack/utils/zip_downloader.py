import urllib
import zipfile
import os
from tqdm import tqdm


def make_zip_downloader(URL : str, file_list=None, resource_name = None):
    """
    This function is used to make a zipfile downloader for data.
    """
    if isinstance(file_list, str):
        file_list = [file_list]
    
    use_source = not (URL.startswith("http://") or URL.startswith("https://"))
    if use_source and URL.startswith("/"):
        URL = URL[1:]

    def DOWNLOAD(path : str, source : str):
        if not source.endswith("/"):
            source = source + "/"
        if use_source:
            remote_url = source + URL
        else:
            remote_url = URL

        if resource_name is None:
            name = os.path.basename(path)
        else:
            name = resource_name
        
        with urllib.request.urlopen(remote_url) as fin:
            CHUNK_SIZE = 4 * 1024
            total_length = int(fin.headers["content-length"])
            with open(path + ".zip", "wb") as ftmp:
                with tqdm(total=total_length, unit="B", desc="Downloading %s" % name, unit_scale=True) as pbar:
                    while True:
                        data = fin.read(CHUNK_SIZE)
                        if len(data) == 0:
                            break
                        ftmp.write(data)
                        pbar.update(len(data))
                ftmp.flush()
                
        zf = zipfile.ZipFile(path + ".zip")
        
        os.makedirs(path, exist_ok=True)
        if file_list is not None:
            for file in file_list:
                zf.extract(file, path)
        else:
            zf.extractall(path)
        zf.close()
        os.remove(path + ".zip")
        return

    return DOWNLOAD
