# base
from .base import Attacker
from .classification import ClassificationAttacker

# classification
from .genetic import GeneticAttacker
from .scpn import SCPNAttacker
from .fd import FDAttacker
from .hotflip import HotFlipAttacker
from .textfooler import TextFoolerAttacker
from .pwws import PWWSAttacker
from .uat import UATAttacker
from .viper import VIPERAttacker
from .deepwordbug import DeepWordBugAttacker
from .gan import GANAttacker
from .textbugger import TextBuggerAttacker
from .pso import PSOAttacker
from .bert_attack import BERTAttacker
from .bae import BAEAttacker
# from .geometry import GEOAttacker  FIXME: cannot import name 'zero_gradients' from 'torch.autograd.gradcheck'
