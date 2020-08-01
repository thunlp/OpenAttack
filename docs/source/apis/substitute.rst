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

DCESSubstitute
----------------

.. autoclass:: OpenAttack.substitutes.DCESSubstitute(OpenAttack.substitutes.CharSubstitute)
    :members:

ECESSubstitute
----------------

.. autoclass:: OpenAttack.substitutes.ECESSubstitute(OpenAttack.substitutes.CharSubstitute)
    :members:

CounterFittedSubstitute
-------------------------

.. autoclass:: OpenAttack.substitutes.CounterFittedSubstitute(OpenAttack.substitutes.EmbedBasedSubstitute)
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

