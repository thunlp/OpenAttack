"""
:type: OpenAttack.utils.RobertaClassifier
:Size: 1.18GB
:Package Requirements:
    * transformers
    * pytorch

Pretrained ROBERTA model on SST-2 dataset. See :py:data:`Dataset.SST` for detail.
"""

from OpenAttack.utils import make_zip_downloader

NAME = "Victim.ROBERTA.SST"

URL = "https://cdn.data.thunlp.org/TAADToolbox/victim/roberta_sst.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    from OpenAttack import HuggingfaceClassifier
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(path, num_labels=2, output_hidden_states=False)
    return HuggingfaceClassifier(model, tokenizer=tokenizer, max_len=100, embedding_layer=model.roberta.embeddings.word_embeddings)