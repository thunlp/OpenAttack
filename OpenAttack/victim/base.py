
import functools
from typing import Union
from .context import AttackContext, AttackContextShadow
from ..exceptions import InvokeLimitExceeded
from .method import VictimMethod
import time

def invoke_decorator(func, method : VictimMethod):
    @functools.wraps(func)
    def invoke_wrapper(self : Victim, *args, **kwargs):
        cnt = method.invoke_count(*args, **kwargs)
        return self.record_invoke(cnt, func, *args, **kwargs)
        
    return invoke_wrapper

class Victim:
    @property
    def TAGS(self):
        return self._method_tags

    def __init_subclass__(cls, invoke_funcs=[], tags=set()):
        for func_name, method in invoke_funcs:
            setattr( cls, func_name, invoke_decorator( getattr(cls, func_name), method ) )
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
    def context(self) -> Union[None, AttackContextShadow]:
        if not hasattr(self, "_Victim__context"):
            return None
        else:
            return AttackContextShadow(self._Victim__context)

    def record_invoke(self, cnt, func, *args, **kwargs):
        
        if hasattr(self, "_Victim__context"):
            need_record = (self._Victim__context is not None) and (not self._Victim__context.inference)
        else:
            need_record = False
        
        if need_record:
            self._Victim__context.inference = True
            if self._Victim__context.invoke_limit is not None and self._Victim__context.invoke + cnt > self._Victim__context.invoke_limit:
                raise InvokeLimitExceeded()
            else:
                self._Victim__context.invoke += cnt
            st = time.time()
        
        # call original function here
        ret = func(self, *args, **kwargs)
        
        if need_record:
            self._Victim__context.inference = False
            self._Victim__context.attacker_time_del +=  time.time() - st
        
        return ret
