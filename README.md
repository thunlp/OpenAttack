<h1 align="center">OpenAttack</h1>
<p align="center">
  <a target="_blank">
    <img src="https://github.com/thunlp/OpenAttack/workflows/Test/badge.svg?branch=master" alt="Github Runner Covergae Status">
  </a>
  <a href="https://openattack.readthedocs.io/" target="_blank">
    <img src="https://readthedocs.org/projects/openattack/badge/?version=latest" alt="ReadTheDoc Status">
  </a>
  <a  href="https://pypi.org/project/OpenAttack/"  target="_blank">
    <img src="https://img.shields.io/pypi/v/OpenAttack?label=OpenAttack" alt="PyPI version">
  </a>
   <a target="_blank">
    <img src="https://img.shields.io/badge/PRs-Welcome-red" alt="PRs are Welcome">
  </a>
<br><br>
  <a href="https://openattack.readthedocs.io/" target="_blank">Documentation</a> • <a href="#features--uses">Features & Uses</a> • <a href="#usage-examples">Usage Examples</a> • <a href="#attack-models">Attack Models</a> • <a href="#toolkit-design">Toolkit Design</a> 
<br>
</p>

OpenAttack is an open-source Python-based textual adversarial attack toolkit, which handles the whole process of textual adversarial attacking, including preprocessing text, accessing the victim model, generating adversarial examples and evaluation. 

![demo](./docs/source/images/demo.gif)

## Features & Uses

OpenAttack has following features:

1. **High usability.** OpenAttack provides easy-to-use APIs that can support the whole process of textual adversarial attacks;
2. **Full coverage of attack model types**. OpenAttack supports sentence-/word-/character-level perturbations and gradient-/score-/decision-based/blind attack models;
3. **Great flexibility and extensibility**. You can easily attack a customized victim model or develop and evaluate a customized attack model;
4. **Comprehensive Evaluation**. OpenAttack can thoroughly evaluate an attack model from attack effectiveness, adversarial example quality and attack efficiency.



OpenAttack has a wide range of uses, including:

1. Providing various handy baselines for attack models; 
2. Comprehensively evaluating attack models using its thorough evaluation metrics; 
3. Assisting in quick development of new attack models with the help of its common attack components; 
4. Evaluating the robustness of a machine learning model against various adversarial attacks; 
5. Conducting adversarial training to improve robustness of a machine learning model by enriching the training data with generated adversarial examples.

## Installation

You can either use `pip` or clone this repo to install OpenAttack.

#### 1. Using pip (recommended)

```bash
pip install OpenAttack
```

#### 2. Cloning this repo

```bash
git clone https://github.com/thunlp/OpenAttack.git
cd OpenAttack
python setup.py install
```



After installation, you can try running `demo.py` to check if OpenAttack works well:

```
python demo.py
```

## Usage Examples

#### Basic: Use Built-in Attacks

OpenAttack builds in some commonly used text classification models such as LSTM and BERT as well as datasets such as [SST](https://nlp.stanford.edu/sentiment/treebank.html) for sentiment analysis and [SNLI](https://nlp.stanford.edu/projects/snli/) for natural language inference. You can effortlessly conduct adversarial attacks against the built-in victim models on the datasets.

The following code snippet shows how to use a genetic algorithm-based attack model ([Alzantot et al., 2018](https://www.aclweb.org/anthology/D18-1316.pdf)) to attack BERT on the SST dataset:

```python
import OpenAttack as oa
# choose a trained victim classification model
victim = oa.DataManager.load("Victim.BERT.SST")
# choose an evaluation dataset 
dataset = oa.DataManager.load("Dataset.SST.sample")
# choose Genetic as the attacker and initialize it with default parameters
attacker = oa.attackers.GeneticAttacker() 
# prepare for attacking
attack_eval = oa.attack_evals.DefaultAttackEval(attacker, victim)
# launch attacks and print attack results 
attack_eval.eval(dataset, visualize=True)
```

#### Advanced: Attack a Customized Victim Model

The following code snippet shows how to use the genetic algorithm-based attack model to attack a customized sentiment analysis model (a statistical model built in NLTK) on SST.

```python
import OpenAttack as oa
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# configure access interface of the customized victim model
class MyClassifier(oa.Classifier):
    def __init__(self):
        self.model = SentimentIntensityAnalyzer()
    # access to the classification probability scores with respect input sentences
    def get_prob(self, input_): 
        rt = []
        for sent in input_:
            rs = self.model.polarity_scores(sent)
            prob = rs["pos"] / (rs["neg"] + rs["pos"])
            rt.append(np.array([1 - prob, prob]))
        return np.array(rt)
# choose the costomized classifier as the victim model
victim = MyClassifier()
# choose an evaluation dataset 
dataset = oa.DataManager.load("Dataset.SST.sample")
# choose Genetic as the attacker and initialize it with default parameters
attacker = oa.attackers.GeneticAttacker()
# prepare for attacking
attack_eval = oa.attack_evals.DefaultAttackEval(attacker, victim)
# launch attacks and print attack results 
attack_eval.eval(dataset, visualize=True)
```

#### Advanced: Design a Customized Attack Model

OpenAttack incorporates many handy components which can be easily assembled into new attack model. 

[Here](./examples/custom_attacker.py) gives an example of how to design a simple attack model which shuffles the tokens in the original sentence.

#### Advanced: Adversarial Training

OpenAttack can easily generate adversarial examples by attacking instances in the training set, which can be added to original training data set to retrain a more robust victim model, i.e., adversarial training. 

[Here](./examples/adversarial_training.py)  gives an example of how to conduct adversarial training with OpenAttack.

#### Advanced: Design a Customized Evaluation Metric

OpenAttack supports designing a customized adversarial attack evaluation metric.

[Here](./examples/custom_eval.py)  gives an example of how to add BLEU score as a customized evaluation metric to evaluate adversarial attacks.

## Attack Models

According to the level of perturbations imposed on original input, textual adversarial attack models can be categorized into sentence-level, word-level, character-level attack models. 

According to the accessibility to the victim model, textual adversarial attack models can be categorized into `gradient`-based, `score`-based, `decision`-based and `blind` attack models.

> [TAADPapers](https://github.com/thunlp/TAADpapers) is a paper list which summarizes almost all the papers concerning textual adversarial attack and defense. You can have a look at this list to find more attack models.

Currently OpenAttack includes 13 typical attack models against text classification models that cover **all** attack types. 

Here is the list of currently involved attack models.

- Sentence-level
  - (SEA) **Semantically Equivalent Adversarial Rules for Debugging NLP Models**. *Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin*. ACL 2018. `decision` [[pdf](https://aclweb.org/anthology/P18-1079)] [[code](https://github.com/marcotcr/sears)]
  - (SCPN) **Adversarial Example Generation with Syntactically Controlled Paraphrase Networks**. *Mohit Iyyer, John Wieting, Kevin Gimpel, Luke Zettlemoyer*. NAACL-HLT 2018. `blind` [[pdf](https://www.aclweb.org/anthology/N18-1170)] [[code&data](https://github.com/miyyer/scpn)]
  - (GAN) **Generating Natural Adversarial Examples**. *Zhengli Zhao, Dheeru Dua, Sameer Singh*. ICLR 2018. `decision` [[pdf](https://arxiv.org/pdf/1710.11342.pdf)] [[code](https://github.com/zhengliz/natural-adversary)]
- Word-level
  - (SememePSO) **Word-level Textual Adversarial Attacking as Combinatorial Optimization**. *Yuan Zang, Fanchao Qi, Chenghao Yang, Zhiyuan Liu, Meng Zhang, Qun Liu and Maosong Sun*. ACL 2020. `score` [[pdf](https://www.aclweb.org/anthology/2020.acl-main.540.pdf)] [[code](https://github.com/thunlp/SememePSO-Attack)]
  - (TextFooler) **Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment**. *Di Jin, Zhijing Jin, Joey Tianyi Zhou, Peter Szolovits*. AAAI-20. `score` [[pdf](https://arxiv.org/pdf/1907.11932v4)] [[code](https://github.com/wqj111186/TextFooler)]
  - (PWWS) **Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency**. *Shuhuai Ren, Yihe Deng, Kun He, Wanxiang Che*. ACL 2019. `score` [[pdf](https://www.aclweb.org/anthology/P19-1103.pdf)] [[code](https://github.com/JHL-HUST/PWWS/)]
  - (Genetic) **Generating Natural Language Adversarial Examples**. *Moustafa Alzantot, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani Srivastava, Kai-Wei Chang*. EMNLP 2018. `score` [[pdf](https://www.aclweb.org/anthology/D18-1316)] [[code](https://github.com/nesl/nlp_adversarial_examples)]
  - (FD) **Crafting Adversarial Input Sequences For Recurrent Neural Networks**. *Nicolas Papernot, Patrick McDaniel, Ananthram Swami, Richard Harang*. MILCOM 2016. `gradient` [[pdf](https://arxiv.org/pdf/1604.08275.pdf)]
- Word/Char-level
  - (UAT) **Universal Adversarial Triggers for Attacking and Analyzing NLP.** *Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, Sameer Singh*. EMNLP-IJCNLP 2019. `gradient` [[pdf](https://arxiv.org/pdf/1908.07125.pdf)] [[code](https://github.com/Eric-Wallace/universal-triggers)] [[website](http://www.ericswallace.com/triggers)]
  - (TextBugger) **TEXTBUGGER: Generating Adversarial Text Against Real-world Applications**. *Jinfeng Li, Shouling Ji, Tianyu Du, Bo Li, Ting Wang*. NDSS 2019. `gradient` `score` [[pdf](https://arxiv.org/pdf/1812.05271.pdf)]
  - (HotFlip) **HotFlip: White-Box Adversarial Examples for Text Classification**. *Javid Ebrahimi, Anyi Rao, Daniel Lowd, Dejing Dou*. ACL 2018. `gradient` [[pdf](https://www.aclweb.org/anthology/P18-2006)] [[code](https://github.com/AnyiRao/WordAdver)]
- Char-level
  - (VIPER) **Text Processing Like Humans Do: Visually Attacking and Shielding NLP Systems**. *Steffen Eger, Gözde Gül ¸Sahin, Andreas Rücklé, Ji-Ung Lee, Claudia Schulz, Mohsen Mesgar, Krishnkant Swarnkar, Edwin Simpson, Iryna Gurevych*. NAACL-HLT 2019. `score` [[pdf](https://www.aclweb.org/anthology/N19-1165)] [[code&data](https://github.com/UKPLab/naacl2019-like-humans-visual-attacks)]
  - (DeepWordBug) **Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers**. *Ji Gao, Jack Lanchantin, Mary Lou Soffa, Yanjun Qi*. IEEE SPW 2018. `score` [[pdf](https://ieeexplore.ieee.org/document/8424632)] [[code](https://github.com/QData/deepWordBug)]



Following table illustrates the comparison of the attack models.

|    Model    |  Accessibility  | Perturbation | Main Idea                                           |
| :---------: | :-------------: | :----------: | :-------------------------------------------------- |
|     SEA     |    Decision     |   Sentence   | Rule-based paraphrasing                             |
|    SCPN     |      Blind      |   Sentence   | Paraphrasing                                        |
|     GAN     |    Decision     |   Sentence   | Text generation by encoder-decoder                  |
|  SememePSO  |      Score      |     Word     | Particle Swarm Optimization-based word substitution |
| TextFooler  |      Score      |     Word     | Greedy word substitution                            |
|    PWWS     |      Score      |     Word     | Greedy word substitution                            |
|   Genetic   |      Score      |     Word     | Genetic algorithm-based word substitution           |
|     FD      |    Gradient     |     Word     | Gradient-based word substitution                    |
| TextBugger  | Gradient, Score |  Word+Char   | Greedy word substitution and character manipulation |
|     UAT     |    Gradient     |  Word, Char  | Gradient-based word or character manipulation       |
|   HotFlip   |    Gradient     |  Word, Char  | Gradient-based word or character substitution       |
|    VIPER    |      Blind      |     Char     | Visually similar character substitution             |
| DeepWordBug |      Score      |     Char     | Greedy character manipulation                       |

## Toolkit Design

Considering the significant distinctions among different attack models, we leave considerable freedom for the skeleton design of attack models, and focus more on streamlining the general processing of adversarial attacking and the common components used in attack models.

OpenAttack has 7 main modules: 

<img src="./docs/source/images/toolkit_framework.png" alt="toolkit_framework" style="zoom:30%;" />

* **TextProcessor**: processing the original text sequence so as to assist attack models in generating adversarial examples.
* **Classifier**: wrapping victim classification models
* **Attacker**: involving various attack models
* **Substitute**: packing different word/character substitution methods which are widely used in word- and character-level attack models.
* **Metric**: providing several adversarial example quality metrics which can serve as either the constraints on the adversarial examples during attacking or evaluation metrics for evaluating adversarial attacks.
* **AttackEval**: evaluating textual adversarial attacks from attack effectiveness, adversarial example quality and attack efficiency.
* **DataManager**: managing all the data as well as saved models that are used in other modules