"""
:type: OpenAttack.utils.AlbertClassifier
:Size: 788.697MB
:Package Requirements:
    * transformers
    * pytorch

Pretrained ALBERT model on AG-4 dataset. See :py:data:`Dataset.AG` for detail.
"""

from OpenAttack.utils import make_zip_downloader

NAME = "Victim.ALBERT.AG"

URL = "/TAADToolbox/victim/albert_ag.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    from OpenAttack.victim.classifiers import TransformersClassifier
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(path, num_labels=5, output_hidden_states=False)
    return TransformersClassifier(model, tokenizer, model.albert.embeddings.word_embeddings)