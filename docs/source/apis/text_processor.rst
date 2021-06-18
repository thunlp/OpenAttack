========================
Text Processors API
========================

Tokenizers
============================


.. autoclass:: OpenAttack.text_process.tokenizer.Tokenizer
    :members: tokenize, detokenize

JiebaTokenizer
----------------

.. autoclass:: OpenAttack.text_process.tokenizer.JiebaTokenizer(OpenAttack.text_process.tokenizer.Tokenizer)
    :members:

PunctTokenizer
----------------

.. autoclass:: OpenAttack.text_process.tokenizer.PunctTokenizer(OpenAttack.text_process.tokenizer.Tokenizer)
    :members:

TransformersTokenizer
-----------------------

.. autoclass:: OpenAttack.text_process.tokenizer.TransformersTokenizer(OpenAttack.text_process.tokenizer.Tokenizer)
    :members:

Lemmatizer
============================


.. autoclass:: OpenAttack.text_process.lemmatizer.Lemmatizer
    :members: lemmatize, delemmatize

WordnetLemmatimer
-------------------

.. autoclass:: OpenAttack.text_process.lemmatizer.WordnetLemmatimer(OpenAttack.text_process.lemmatizer.Lemmatizer)
    :members:

ConstituencyParser
============================


.. autoclass:: OpenAttack.text_process.constituency_parser.ConstituencyParser
    :members: __call__

StanfordParser
----------------

.. autoclass:: OpenAttack.text_process.constituency_parser.StanfordParser(OpenAttack.text_process.constituency_parser.ConstituencyParser)
    :members:

