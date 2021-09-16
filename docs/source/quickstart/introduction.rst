====================
Introduction
====================

OpenAttack is an open-source Python-based textual adversarial attack toolkit, which handles the whole process of textual adversarial attacking, including preprocessing text, accessing the victim model, generating adversarial examples and evaluation. 


Features & Uses
====================

OpenAttack has the following features:
----------------------------------------------

⭐️ **Support for all attack types**. OpenAttack supports all types of attacks including sentence-/word-/character-level perturbations and gradient-/score-/decision-based/blind attack models;

⭐️ **Multilinguality**. OpenAttack supports English and Chinese now. Its extensible design enables quick support for more languages;

⭐️ **Parallel processing**. OpenAttack provides support for multi-process running of attack models to improve attack efficiency;

⭐️ **Compatibility with 🤗 Hugging Face**. OpenAttack is fully integrated with 🤗  `Transformers <https://github.com/huggingface/transformers>`__ and `Datasets <https://github.com/huggingface/datasets>`__ libraries;

⭐️ **Great extensibility**. You can easily attack a customized <u>victim model</u> on any customized <u>dataset</u> or develop and evaluate a customized <u>attack model</u>.



OpenAttack has a wide range of uses, including:
-------------------------------------------------------------

✅ Providing various handy **baselines** for attack models; 

✅ Comprehensively **evaluating** attack models using its thorough evaluation metrics; 

✅ Assisting in quick development of **new attack models** with the help of its common attack components; 

✅ Evaluating the **robustness** of a machine learning model against various adversarial attacks; 

✅ Conducting **adversarial training** to improve robustness of a machine learning model by enriching the training data with generated adversarial examples.

