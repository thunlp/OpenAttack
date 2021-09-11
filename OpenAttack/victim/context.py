import time

class AttackContext:
    def __init__(self, data, invoke_limit) -> None:
        self.input = data
        self.invoke = 0
        self.invoke_limit = invoke_limit
        self.attacker_start = time.time()
        self.attacker_time_del = 0
        self.inference = False
    
    @property
    def attack_time(self):
        return time.time() - self.attacker_start - self.attacker_time_del

    def __setattr__(self, name, value) -> None:
        if name in ["invoke"] and hasattr(self, name):
            if getattr(self, name) > value:
                raise RuntimeError("Invalid access")
            else:
                super().__setattr__(name, value)    
        else:
            super().__setattr__(name, value)

class AttackContextShadow:
    invoke : int
    invoke_limit : int
    attacker_start : float
    attacker_time_del : int
    attacker_time : int
    input : dict

    def __init__(self, ctx) -> None:
        self.__ctx = ctx
    
    def __setattr__(self, name: str, value) -> None:
        if name in ["invoke", "invoke_limit", "attacker_start", "attacker_time_del"]:
            raise TypeError("'AttackContext' object does not support item assignment")
        elif name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self.__ctx, name, value)
    
    def __getattribute__(self, name: str):
        if name in ["attacker_start", "attacker_time_del"]:
            raise AttributeError("'AttackContext' object has no attribute '%s'" % name)
        elif name.startswith("_"):
            return super().__getattribute__(name)
        else:
            return getattr(self.__ctx, name)
    
    def __delattr__(self, name: str) -> None:
        if name in ["invoke", "invoke_limit", "attacker_start", "attacker_time_del", "attacker_time"]:
            raise AttributeError("%s" % name)
        elif name.startswith("_"):
            super().__delattr__(name)
        else:
            delattr(self.__ctx, name)