========================
AttackEvals API
========================

AttackEval
----------------

.. autoclass:: OpenAttack.AttackEval
    :members: __init__, eval, eval_results

------------------------------------

AttackEvalBase
----------------

.. autoclass:: OpenAttack.attack_evals.AttackEvalBase(OpenAttack.AttackEval)
    :members: __init__, measure, update, get_result, clear, eval, eval_results

DefaultAttackEval
-------------------

.. autoclass:: OpenAttack.attack_evals.DefaultAttackEval(OpenAttack.AttackEval)
    :members: __init__, measure, update, get_result, clear, eval, eval_results

InvokeLimitedAttackEval
-------------------------

.. autoclass:: OpenAttack.attack_evals.InvokeLimitedAttackEval(OpenAttack.AttackEval)
    :members: __init__, measure, update, get_result, clear, eval, eval_results

