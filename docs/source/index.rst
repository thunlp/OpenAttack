OpenAttack
=======================================

OpenAttack is an open-source Python-based textual adversarial attack toolkit, 
which handles the whole process of textual adversarial attacking, including 
preprocessing text, accessing the victim model, generating adversarial examples 
and evaluation.

------------------
Uses
------------------
OpenAttack has a wide range of uses, including:

1. Providing various handy baselines for attack models;
2. Comprehensively evaluating attack models using its thorough evaluation metrics;
3. Assisting in quick development of new attack models with the help of its common attack components;
4. Evaluating the robustness of a machine learning model against various adversarial attacks;
5. Conducting adversarial training to improve robustness of a machine learning model by enriching the training data with generated adversarial examples.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   Installation <quickstart/installation>
   Introduction <quickstart/introduction>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Examples

   Example 1: Workflow <examples/example1>
   Example 2: Classifier <examples/example2>
   Example 3: Attacker <examples/example3>
   Example 4: AttackEval <examples/example4>
   
.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Module Reference

   TextProcessor <module/text_processor>
   Substitute <module/substitute>
   Classifier <module/classifier>
   Attacker <module/attacker>
   AttackEval <module/attack_eval>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: APIs

   DataManager <apis/data_manager>
   Attacker <apis/attacker>
   Substitute <apis/substitute>
   Metric <apis/metric>
   TextProcessor <apis/text_processor>
   AttackEval <apis/attack_eval>
   Exceptions <apis/exceptions>
   data <apis/data>
   utils <apis/utils>

Indices and tables
==================

* :ref:`search`