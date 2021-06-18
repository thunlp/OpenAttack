"""
:type: OpenAttack.utils.XlnetClassifier
:Size: 1.25GB
:Package Requirements:
    * transformers
    * pytorch

Pretrained XLNET model on MNLI dataset. See :py:data:`Dataset.MNLI` for detail.
"""

from OpenAttack.utils import make_zip_downloader

NAME = "Victim.XLNET.MNLI"

URL = "/TAADToolbox/victim/xlnet_mnli.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(path, num_labels=4, output_hidden_states=False)
    
    from OpenAttack.victim.classifiers import TransformersClassifier
    return TransformersClassifier(model, tokenizer, model.transformer.word_embedding)