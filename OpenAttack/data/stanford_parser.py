"""
:type: function
:Size: 603.979MB
:Package Requirements: * **Java**

Model files for Stanford Parser.
`[page] <https://nlp.stanford.edu/software/lex-parser.shtml>`__
"""
from OpenAttack.utils import make_zip_downloader
import os

NAME = "TProcess.StanfordParser"

URL = "/TAADToolbox/stanford_parser_small.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    return (
        __import__("nltk.parse.stanford")
        .parse.stanford.StanfordParser(
            path_to_jar=os.path.join(path, "stanford-parser.jar"),
            path_to_models_jar=os.path.join(path, "stanford-parser-3.9.2-models.jar"),
            model_path=os.path.join(path, "englishPCFG.ser.gz"),
        )
        .raw_parse
    )
