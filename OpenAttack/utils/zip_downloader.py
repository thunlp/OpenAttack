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
        try:
            with urllib.request.urlopen(URL) as f:
                zf = zipfile.ZipFile(io.BytesIO(f.read()))
        except OverflowError:
            with urllib.request.urlopen(URL) as f:
                CHUNK_SIZE = 1024 * 1024 * 10
                ftmp = open(path + ".zip", "wb")
                while True:
                    data = f.read(CHUNK_SIZE)
                    ftmp.write(data)
                    if len(data) == 0:
                        break
                ftmp.flush()
                ftmp.close()
                zf = zipfile.ZipFile(path + ".zip")
        
        os.makedirs(path, exist_ok=True)
        if file_list is not None:
            for file in file_list:
                zf.extract(file, path)
        else:
            zf.extractall(path)
        zf.close()
        return

    return DOWNLOAD
