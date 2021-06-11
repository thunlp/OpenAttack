import functools
from typing import Set
from .context import AttackContext, AttackContextShadow
from ..exceptions import InvokeLimitExceeded
from ..tags import Tag
import time

def invoke_decorator(func):
    @functools.wraps(func)
    def invoke_wrapper(self : Victim, *args, **kwargs):
        return self.record_invoke(func, *args, **kwargs)
        
    return invoke_wrapper

class Victim:
    @property
    def TAGS(self):
        return self._method_tags

    def __init_subclass__(cls, invoke_funcs=[], tags=set()):
        for func_name in invoke_funcs:
            setattr( cls, func_name, invoke_decorator( getattr(cls, func_name) ) )
        cls._method_tags = set(tags)
    
    @property
    def supported_language(self):
        for tag in self.TAGS:
            if tag.type == "lang":
                return tag
        return None
    
    def set_context(self, data, invoke_limit):
        self._Victim__context = AttackContext(data, invoke_limit)

    def clear_context(self):
        self._Victim__context = None
    
    @property
    def context(self):
        if not hasattr(self, "_Victim__context"):
            return None
        else:
            return AttackContextShadow(self._Victim__context)

    def record_invoke(self, func, *args, **kwargs):
        
        if hasattr(self, "_Victim__context"):
            need_record = (self._Victim__context is not None) and (not self._Victim__context.inference)
        else:
            need_record = False
        
        if need_record:
            self._Victim__context.inference = True
            if self._Victim__context.invoke_limit is not None and self._Victim__context.invoke >= self._Victim__context.invoke_limit:
                raise InvokeLimitExceeded()
            else:
                self._Victim__context.invoke += 1
            st = time.time()
        
        # call original function here
        ret = func(self, *args, **kwargs)
        
        if need_record:
            self._Victim__context.inference = False
            self._Victim__context.attacker_time_del +=  time.time() - st
        
        return ret
