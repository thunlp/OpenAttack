"""
:type: OpenAttack.utils.AlbertClassifier
:Size: 788.662MB
:Package Requirements:
    * transformers
    * pytorch

Pretrained ALBERT model on IMDB dataset. See :py:data:`Dataset.IMDB` for detail.
"""

from OpenAttack.utils import make_zip_downloader

NAME = "Victim.ALBERT.IMDB"

URL = "/TAADToolbox/victim/albert_imdb.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(path, num_labels=2, output_hidden_states=False)
    
    from OpenAttack.victim.classifiers import TransformersClassifier
    return TransformersClassifier(model, tokenizer, model.albert.embeddings.word_embeddings)