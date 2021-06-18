"""
:type: function
:Size: 21.164MB
:Package Requirements: * **Java**

Model files for Stanford NER tagger.
`[page] <https://nlp.stanford.edu/software/CRF-NER.html>`__
"""
from OpenAttack.utils import make_zip_downloader
import os

NAME = "TProcess.StanfordNER"

URL = "/TAADToolbox/stanford_ner_small.zip"
DOWNLOAD = make_zip_downloader(URL)


def LOAD(path):
    return (
        __import__("nltk")
        .StanfordNERTagger(
            model_filename=os.path.join(path, "english.muc.7class.distsim.crf.ser.gz"),
            path_to_jar=os.path.join(path, "stanford-ner.jar"),
        )
        .tag
    )
