"""
:type: OpenAttack.utils.BertClassifier
:Size: 992.75 MB
:Package Requirements:
    * transformers
    * pytorch

Pretrained BERT model on Amazon Reviews (Chinese) dataset.
"""

from OpenAttack.utils import make_zip_downloader
import os

NAME = "Victim.BERT.AMAZON_ZH"

URL = "/TAADToolbox/victim/bert_amazon_reviews_zh.zip"
DOWNLOAD = make_zip_downloader(URL)

def LOAD(path):
    import transformers
    tokenizer = transformers.BertTokenizer.from_pretrained(path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(path, num_labels=5, output_hidden_states=False)

    from OpenAttack.victim.classifiers import TransformersClassifier
    return TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings, lang="chinese")
