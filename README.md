# OpenAttack

![Test](https://github.com/Fanchao-Qi/TAADToolbox/workflows/Test/badge.svg?branch=master)

OpenAttack is an open-source Python-based textual adversarial attack toolkit, which handles the whole process of textual adversarial attacking, including preprocessing text, accessing the victim model, generating adversarial examples and evaluation. 

## Uses

OpenAttack has a wide range of uses, including:

1. Providing various handy baselines for attack models; 
2. Comprehensively evaluating attack models using its thorough evaluation metrics; 
3. Assisting in quick development of new attack models with the help of its common attack components; 
4. Evaluating the robustness of a machine learning model against various adversarial attacks; 
5. Conducting adversarial training to improve robustness of a machine learning model by enriching the training data with generated adversarial examples.



## Installation

You can just down the whole repo for installation.

## Usage Examples

#### Use Built-in Attacks and Evaluation

OpenAttack builds in some commonly used text classification models such as LSTM and BERT as well as datasets such as SST for sentiment analysis and SNLI for natural language inference.
You can use the built-in victim models and datasets to quickly conduct adversarial attacks.

The following code snippet shows how to use Genetic to attack BERT on the SST dataset:

```python
import OpenAttack as oa
# choose trained victim model
victim = oa.DataManager.load("Victim.BERT.SST")
# choose evaluation dataset 
dataset = oa.DataManager.load("Dataset.SST.sample")[:10]
# choose Genetic as the attacker
attacker = oa.attackers.GeneticAttacker() 
# prepare for attacking
attack_eval = oa.attack_evals.DefaultAttackEval(attacker, victim)
# launch attacks and print attack results 
attack_eval.eval(dataset, visualize=True)
```

#### Attack a Customized Victim Model

The following code snippet shows how to use Genetic to attack a customized sentiment analysis model (a statistical model built in NLTK) on SST.

```python
import OpenAttack as oa
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# configure access interface of customized model
class MyClassifier(oa.Classifier):
    def __init__(self):
        self.model = SentimentIntensityAnalyzer()
    def get_prob(self, input_):
        rt = []
        for sent in input_:
            rs = self.model.polarity_scores(sent)
            prob = rs["pos"] / (rs["neg"] + rs["pos"])
            rt.append(np.array([1 - prob, prob]))
        return np.array(rt)
# choose evaluation dataset 
dataset = oa.load("Dataset.SST.sample")[:10]
# choose the costomized classifier
clsf = MyClassifier()
# choose Genetic as the attack model 
attacker = oa.attackers.GeneticAttacker()
# prepare for attacking
attack_eval = oa.attack_evals.DefaultAttackEval(attacker, clsf)
# launch attacks and print attack results 
attack_eval.eval(dataset, visualize=True)
```



## Attack Models

According to the level of perturbations imposed on original input, textual adversarial attack models can be categorized into sentence-level, word-level, character-level attack models. 

According to the accessibility to the victim model, textual adversarial attack models can be categorized into `gradient`-based, `score`-based, `decision`-based and `blind` attack models.

Currently OpenAttack includes 13 typical attack models against text classification models that cover all attack types. 

Here is the list of currently involved attack models.

- Sentence-level
  - **Semantically Equivalent Adversarial Rules for Debugging NLP Models**. *Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin*. ACL 2018. `decision` [[pdf](https://aclweb.org/anthology/P18-1079)] [[code](https://github.com/marcotcr/sears)]
  - **Adversarial Example Generation with Syntactically Controlled Paraphrase Networks**. *Mohit Iyyer, John Wieting, Kevin Gimpel, Luke Zettlemoyer*. NAACL-HLT 2018. `blind` [[pdf](https://www.aclweb.org/anthology/N18-1170)] [[code&data](https://github.com/miyyer/scpn)]
  - **Generating Natural Adversarial Examples**. *Zhengli Zhao, Dheeru Dua, Sameer Singh*. ICLR 2018. `decision` [[pdf](https://arxiv.org/pdf/1710.11342.pdf)] [[code](https://github.com/zhengliz/natural-adversary)]
- Word-level
  - **Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment**. *Di Jin, Zhijing Jin, Joey Tianyi Zhou, Peter Szolovits*. AAAI-20. `score` [[pdf](https://arxiv.org/pdf/1907.11932v4)] [[code](https://github.com/wqj111186/TextFooler)]
  - **Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency**. *Shuhuai Ren, Yihe Deng, Kun He, Wanxiang Che*. ACL 2019. `score` [[pdf](https://www.aclweb.org/anthology/P19-1103.pdf)] [[code](https://github.com/JHL-HUST/PWWS/)]
  - **Generating Natural Language Adversarial Examples**. *Moustafa Alzantot, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani Srivastava, Kai-Wei Chang*. EMNLP 2018. `score` [[pdf](https://www.aclweb.org/anthology/D18-1316)] [[code](https://github.com/nesl/nlp_adversarial_examples)]
  - **Crafting Adversarial Input Sequences For Recurrent Neural Networks**. *Nicolas Papernot, Patrick McDaniel, Ananthram Swami, Richard Harang*. MILCOM 2016. `gradient` [[pdf](https://arxiv.org/pdf/1604.08275.pdf)]
- Char-level
  - **Text Processing Like Humans Do: Visually Attacking and Shielding NLP Systems**. *Steffen Eger, Gözde Gül ¸Sahin, Andreas Rücklé, Ji-Ung Lee, Claudia Schulz, Mohsen Mesgar, Krishnkant Swarnkar, Edwin Simpson, Iryna Gurevych*. NAACL-HLT 2019. `score` [[pdf](https://www.aclweb.org/anthology/N19-1165)] [[code&data](https://github.com/UKPLab/naacl2019-like-humans-visual-attacks)]
  - **Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers**. *Ji Gao, Jack Lanchantin, Mary Lou Soffa, Yanjun Qi*. IEEE SPW 2018. `score` [[pdf](https://ieeexplore.ieee.org/document/8424632)] [[code](https://github.com/QData/deepWordBug)]
- Word/Char-level
  - **Universal Adversarial Triggers for Attacking and Analyzing NLP.** *Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, Sameer Singh*. EMNLP-IJCNLP 2019. `gradient` [[pdf](https://arxiv.org/pdf/1908.07125.pdf)] [[code](https://github.com/Eric-Wallace/universal-triggers)] [[website](http://www.ericswallace.com/triggers)]
  - **TEXTBUGGER: Generating Adversarial Text Against Real-world Applications**. *Jinfeng Li, Shouling Ji, Tianyu Du, Bo Li, Ting Wang*. NDSS 2019. `gradient` `score` [[pdf](https://arxiv.org/pdf/1812.05271.pdf)]
  - **HotFlip: White-Box Adversarial Examples for Text Classification**. *Javid Ebrahimi, Anyi Rao, Daniel Lowd, Dejing Dou*. ACL 2018. `gradient` [[pdf](https://www.aclweb.org/anthology/P18-2006)] [[code](https://github.com/AnyiRao/WordAdver)]