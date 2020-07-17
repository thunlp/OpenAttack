import urllib
import zipfile
import os
import io


def make_zip_downloader(URL, file_list=None):
    """
    This function is used to make a zipfile downloader for data.
    """
    if isinstance(file_list, str):
        file_list = [file_list]

    def DOWNLOAD(path):
        with urllib.request.urlopen(URL) as f:
            zf = zipfile.ZipFile(io.BytesIO(f.read()))
            os.makedirs(path, exist_ok=True)
            if file_list is not None:
                for file in file_list:
                    zf.extract(file, path)
            else:
                zf.extractall(path)
        return

    return DOWNLOAD
