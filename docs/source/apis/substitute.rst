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

CounterFittedSubstitute
-------------------------

.. autoclass:: OpenAttack.substitutes.CounterFittedSubstitute(OpenAttack.substitutes.EmbedBasedSubstitute)
    :members:

DcesSubstitute
----------------

.. autoclass:: OpenAttack.substitutes.DcesSubstitute(OpenAttack.substitutes.CharSubstitute)
    :members:

EcesSubstitute
----------------

.. autoclass:: OpenAttack.substitutes.EcesSubstitute(OpenAttack.substitutes.CharSubstitute)
    :members:

EmbedBasedSubstitute
----------------------

.. autoclass:: OpenAttack.substitutes.EmbedBasedSubstitute(OpenAttack.substitutes.WordSubstitute)
    :members:

HowNetSubstitute
------------------

.. autoclass:: OpenAttack.substitutes.HowNetSubstitute(OpenAttack.substitutes.WordSubstitute)
    :members:

WordNetSubstitute
-------------------

.. autoclass:: OpenAttack.substitutes.WordNetSubstitute(OpenAttack.substitutes.WordSubstitute)
    :members:

