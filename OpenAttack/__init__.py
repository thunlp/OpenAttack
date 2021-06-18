# data manager
from .data_manager import DataManager

# attacker
from . import attackers
from .attackers import Attacker, ClassificationAttacker

# victim
from . import victim
from .victim import classifiers
from .victim import Victim
from .victim.classifiers import Classifier

# metrics
from . import metric
from .metric import AttackMetric

# attack_eval
from .attack_eval import AttackEval

# attack_assist
from .attack_assist import goal, substitute, word_embedding, filter_words

# exception
from . import exceptions
from .exception import AttackException

# utils
from . import utils

download = DataManager.download
load = DataManager.load
loadAttackAssist = DataManager.loadAttackAssist
loadVictim = DataManager.loadVictim
loadTProcess = DataManager.loadTProcess

from .version import VERSION as __version__