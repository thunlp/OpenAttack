========================
AttackEvals API
========================

AttackEvalBase
-----------------

.. autoclass:: OpenAttack.attack_evals.AttackEvalBase(OpenAttack.AttackEval)
    :members: __init__, measure, update, get_result, clear

DefaultAttackEval
-------------------

.. autoclass:: OpenAttack.attack_evals.DefaultAttackEval(OpenAttack.attack_evals.AttackEvalBase)
    :members: __init__, measure, update, get_result, clear, eval, eval_results

InvokeLimitedAttackEval
-------------------------

.. autoclass:: OpenAttack.attack_evals.InvokeLimitedAttackEval(OpenAttack.attack_evals.DefaultAttackEval)
    :members: __init__, measure, update, get_result, clear, eval, eval_results
