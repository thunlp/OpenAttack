"""
:type: OpenHowNet.HowNetDict
:Size: unknown
:Package Requirements: * **OpenHowNet**

OpenHowNet.
`[code] <https://github.com/thunlp/OpenHowNet>`__
`[page] <https://openhownet.thunlp.org/>`__
"""

NAME = "AttackAssist.HowNet"


def DOWNLOAD(path):
    OpenHowNet = __import__("OpenHowNet")
    OpenHowNet.download()
    open(path, "w").write("ok")


def LOAD(path):
    if open(path, "r").read() == "ok":
        return __import__("OpenHowNet").HowNetDict()
    else:
        return None
