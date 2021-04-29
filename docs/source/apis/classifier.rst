===================
Classifiers API
===================

Classifier
-----------------

.. autoclass:: OpenAttack.Classifier
    :members:

------------------------------------

HuggingfaceClassifier
-----------------------

.. autoclass:: OpenAttack.classifiers.HuggingfaceClassifier(OpenAttack.Classifier)
    :members: __init__

PytorchClassifier
-------------------

.. autoclass:: OpenAttack.classifiers.PytorchClassifier(OpenAttack.Classifier)
    :members: __init__, to

TensorflowClassifier
----------------------

.. autoclass:: OpenAttack.classifiers.TensorflowClassifier(OpenAttack.Classifier)
    :members: __init__, to

