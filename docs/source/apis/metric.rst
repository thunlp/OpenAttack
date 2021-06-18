==================
Metric API
==================

Attacker Metrics
==================

.. autoclass:: OpenAttack.metric.AttackMetric
    :members:


BLEU
------

.. autoclass:: OpenAttack.metric.BLEU
    :members: __init__, calc_score
    :exclude-members: TAGS

GPT2LM
--------

.. autoclass:: OpenAttack.metric.GPT2LM
    :members: __init__
    :exclude-members: TAGS

GPT2LMChinese
---------------

.. autoclass:: OpenAttack.metric.GPT2LMChinese
    :members: __init__
    :exclude-members: TAGS

JaccardChar
-------------

.. autoclass:: OpenAttack.metric.JaccardChar
    :members: __init__, calc_score
    :exclude-members: TAGS

JaccardWord
-------------

.. autoclass:: OpenAttack.metric.JaccardWord
    :members: __init__, calc_score
    :exclude-members: TAGS

LanguageTool
--------------

.. autoclass:: OpenAttack.metric.LanguageTool
    :members: __init__
    :exclude-members: TAGS

LanguageToolChinese
---------------------

.. autoclass:: OpenAttack.metric.LanguageToolChinese
    :members: __init__
    :exclude-members: TAGS

Levenshtein
-------------

.. autoclass:: OpenAttack.metric.Levenshtein
    :members: __init__, calc_score
    :exclude-members: TAGS

Modification
--------------

.. autoclass:: OpenAttack.metric.Modification
    :members: __init__, calc_score
    :exclude-members: TAGS

SentenceSim
-------------

.. autoclass:: OpenAttack.metric.SentenceSim
    :members: __init__, calc_score
    :exclude-members: TAGS

UniversalSentenceEncoder
--------------------------

.. autoclass:: OpenAttack.metric.UniversalSentenceEncoder
    :members: __init__, calc_score
    :exclude-members: TAGS

Metrics Selector
=======================

.. autoclass:: OpenAttack.metric.MetricSelector
    :members:


EditDistance
--------------

.. autoclass:: OpenAttack.metric.EditDistance
    :members: 
    :exclude-members: TAGS

Fluency
---------

.. autoclass:: OpenAttack.metric.Fluency
    :members: 
    :exclude-members: TAGS

GrammaticalErrors
-------------------

.. autoclass:: OpenAttack.metric.GrammaticalErrors
    :members: 
    :exclude-members: TAGS

ModificationRate
------------------

.. autoclass:: OpenAttack.metric.ModificationRate
    :members: 
    :exclude-members: TAGS

SemanticSimilarity
--------------------

.. autoclass:: OpenAttack.metric.SemanticSimilarity
    :members: 
    :exclude-members: TAGS

