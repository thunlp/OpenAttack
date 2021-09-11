from typing import Callable, Dict, List, Tuple
import numpy as np
from ..base import Victim
from .methods import *
from ...tags import Tag, TAG_Classification

CLASSIFIER_METHODS : Dict[str, VictimMethod] = {
    "get_pred": GetPredict(),
    "get_prob": GetProbability(),
    "get_grad": GetGradient(),
    "get_embedding": GetEmbedding()
}

class Classifier(Victim):
    """
    Classifier is the base class of all classifiers.
    """
    get_pred : Callable[[List[str]], np.ndarray]
    get_prob : Callable[[List[str]], np.ndarray]
    get_grad : Callable[[List[str]], Tuple[np.ndarray, np.ndarray]]
    def __init_subclass__(cls):
        invoke_funcs = []
        tags = [ TAG_Classification ]

        for func_name in CLASSIFIER_METHODS.keys():
            if hasattr(cls, func_name):
                invoke_funcs.append((func_name, CLASSIFIER_METHODS[func_name]))
                tags.append( Tag(func_name, "victim") )
                setattr(cls, func_name, CLASSIFIER_METHODS[func_name].method_decorator( getattr(cls, func_name) ) )
        
        super().__init_subclass__(invoke_funcs, tags)

        