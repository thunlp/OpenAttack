=======================================
OpenAttack
=======================================

topic-trees
========================

.. toctree::
   Home <self>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   Introduction <quickstart/introduction>
   Installation <quickstart/installation>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Examples

   Example 1: Basic Usage <examples/example1>
   Example 2: Customized Victim Model <examples/example2>
   Example 3: Customized Attack Model <examples/example3>
   Example 4: Customized Measurement <examples/example4>
   Example 5: Adversarial Training <examples/example5>
   Example 6: Attacks on Chinese Dataset <examples/example6>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Modules

   DataManager <apis/data_manager>
   Attacker <apis/attacker>
   Substitute <apis/substitute>
   Metric <apis/metric>
   TextProcessor <apis/text_processor>
   AttackEval <apis/attack_eval>
   Victim <apis/victim>
   utils <apis/utils>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Data

   AttackAssist <data/AttackAssist>
   TProcess <data/TProcess>
   Victim <data/Victim>

project-slogans
========================

.. topic:: project slogan short
    :class: project-slogan-short

    An Open-Source Package for Textual Adversarial Attack. 

.. topic:: project slogan long
    :class: project-slogan-long

    OpenAttack is an open-source Python-based textual adversarial attack toolkit, which handles the whole process of textual adversarial attacking, including preprocessing text, accessing the victim model, generating adversarial examples and evaluation.

.. topic:: install link
    :class: link-button

    :doc:`Install</quickstart/installation>`

.. topic:: quickstart link
    :class: link-button

    :doc:`Quick Start</quickstart/introduction>`

project-features
========================

.. topic:: project-feature-1
    :class: project-feature

    .. image:: images/全部.svg

    All-type Support 

        OpenAttack supports all types of attacks including sentence-/word-/character-level perturbations and gradient-/score-/decision-based/blind attack models;

.. topic:: project-feature-2
    :class: project-feature

    .. image:: images/多语言.svg

    Multilinguality

        OpenAttack supports English and Chinese now. Its extensible design enables quick support for more languages


.. topic:: project-feature-3
    :class: project-feature

    .. image:: images/并行数据挖掘.svg

    Parallel processing 

        OpenAttack provides support for multi-process running of attack models to improve attack efficiency

.. topic:: project-feature-4
    :class: project-feature

    .. image:: images/huggingface.png

    Compatibility with 🤗

        OpenAttack is fully integrated with 🤗 `Transformers <https://github.com/huggingface/transformers>`__ and `Datasets <https://github.com/huggingface/datasets>`__ libraries;

.. topic:: project-feature-5
    :class: project-feature

    .. image:: images/可扩展性强.svg

    Extensibility

        You can easily attack a customized victim model on any customized dataset or develop and evaluate a customized attack model.


data-results
========================

Uses
--------------

.. topic:: data result list
    :class: data-result

    .. container::

        .. image:: images/Align_baseline+row.svg

        Attack Baseline

    .. container::

        .. image:: images/攻击.svg

        Attack Evaluation 

    .. container::

        .. image:: images/开发套件.svg

        Develop New Attack Models

    .. container::

        .. image:: images/盾牌.svg

        Evaluate Robustness

    .. container::

        .. image:: images/训练.svg

        Adversarial Training

