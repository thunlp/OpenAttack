"""
:type: OpenAttack.utils.AlbertClassifier
:Size: 788.66MB
:Package Requirements:
    * transformers
    * pytorch

Pretrained ALBERT model on SST-2 dataset. See :py:data:`Dataset.SST` for detail.
"""

from OpenAttack.utils import make_zip_downloader

NAME = "Victim.ALBERT.SST"

URL = "/TAADToolbox/victim/albert_sst.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(path, num_labels=2, output_hidden_states=False)
    
    from OpenAttack.victim.classifiers import TransformersClassifier
    return TransformersClassifier(model, tokenizer, model.albert.embeddings.word_embeddings)