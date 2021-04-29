======================
Substitutes API
======================

WordSubstitute
----------------

.. autoclass:: OpenAttack.substitutes.base.WordSubstitute
    :members: __call__

CharSubstitute
----------------

.. autoclass:: OpenAttack.substitutes.base.CharSubstitute
    :members: __call__

------------------------------------

EmbedBasedSubstitute
----------------------

.. autoclass:: OpenAttack.substitutes.EmbedBasedSubstitute(OpenAttack.substitutes.WordSubstitute)
    :members:

ChineseFYHCharSubstitute
--------------------------

.. autoclass:: OpenAttack.substitutes.ChineseFYHCharSubstitute(OpenAttack.substitutes.CharSubstitute)
    :members:

ChineseHowNetSubstitute
-------------------------

.. autoclass:: OpenAttack.substitutes.ChineseHowNetSubstitute(OpenAttack.substitutes.WordSubstitute)
    :members:

ChineseSimCharSubstitute
--------------------------

.. autoclass:: OpenAttack.substitutes.ChineseSimCharSubstitute(OpenAttack.substitutes.CharSubstitute)
    :members:

ChineseWord2VecSubstitute
---------------------------

.. autoclass:: OpenAttack.substitutes.ChineseWord2VecSubstitute(OpenAttack.substitutes.EmbedBasedSubstitute)
    :members:

ChineseWordNetSubstitute
--------------------------

.. autoclass:: OpenAttack.substitutes.ChineseWordNetSubstitute(OpenAttack.substitutes.WordSubstitute)
    :members:

CounterFittedSubstitute
-------------------------

.. autoclass:: OpenAttack.substitutes.CounterFittedSubstitute(OpenAttack.substitutes.EmbedBasedSubstitute)
    :members:

DCESSubstitute
----------------

.. autoclass:: OpenAttack.substitutes.DCESSubstitute(OpenAttack.substitutes.CharSubstitute)
    :members:

ECESSubstitute
----------------

.. autoclass:: OpenAttack.substitutes.ECESSubstitute(OpenAttack.substitutes.CharSubstitute)
    :members:

ChineseCiLinSubstitute
------------------------

.. autoclass:: OpenAttack.substitutes.ChineseCiLinSubstitute(OpenAttack.substitutes.WordSubstitute)
    :members:

GloveSubstitute
-----------------

.. autoclass:: OpenAttack.substitutes.GloveSubstitute(OpenAttack.substitutes.EmbedBasedSubstitute)
    :members:

HowNetSubstitute
------------------

.. autoclass:: OpenAttack.substitutes.HowNetSubstitute(OpenAttack.substitutes.WordSubstitute)
    :members:

Word2VecSubstitute
--------------------

.. autoclass:: OpenAttack.substitutes.Word2VecSubstitute(OpenAttack.substitutes.EmbedBasedSubstitute)
    :members:

WordNetSubstitute
-------------------

.. autoclass:: OpenAttack.substitutes.WordNetSubstitute(OpenAttack.substitutes.WordSubstitute)
    :members:

