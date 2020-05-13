NAME = "HOWNET"


def DOWNLOAD(path):
    OpenHowNet = __import__("OpenHowNet")
    OpenHowNet.download()
    open(path, "w").write("ok")


def LOAD(path):
    if open(path, "r").read() == "ok":
        return __import__("OpenHowNet").HowNetDict()
    else:
        return None
