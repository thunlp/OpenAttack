"""
:type: OpenAttack.utils.BertClassifier
:Size: 1.23GB
:Package Requirements:
    * transformers
    * pytorch

Pretrained BERT model on MNLI dataset. See :py:data:`Dataset.MNLI` for detail.
"""

from OpenAttack.utils import make_zip_downloader

NAME = "Victim.BERT.MNLI"

URL = "/TAADToolbox/victim/bert_mnli.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(path, num_labels=2, output_hidden_states=False)
    
    from OpenAttack.victim.classifiers import TransformersClassifier
    return TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)
