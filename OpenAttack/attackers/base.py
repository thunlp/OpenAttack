from typing import Any, Set
from ..victim.base import Victim
from ..attack_assist.goal.base import AttackGoal
from ..tags import Tag

class Attacker:
    """
    The base class of all attackers.
    """


    TAGS : Set[Tag] = set()

    def __call__(self, victim : Victim, input_ : Any):
        raise NotImplementedError()
    
    def _victim_check(self, victim : Victim):
        lang = victim.supported_language
        if lang is not None and lang not in self.TAGS:
            available_langs = []
            for it in self.TAGS:
                if it.type == "lang":
                    available_langs.append(it.name)
            raise RuntimeError("Victim supports language `%s` but `%s` expected." % (lang.name, available_langs))
        
        for tag in self.TAGS:
            if tag.type == "victim":
                if tag not in victim.TAGS:
                    raise AttributeError("`%s` needs victim to support `%s` method" % (self.__class__.__name__, tag.name))

    def attack(self, victim : Victim, input_ : Any, goal : AttackGoal):
        raise NotImplementedError()