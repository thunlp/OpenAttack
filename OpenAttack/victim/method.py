import functools

class VictimMethod:
    def invoke_count(self, *args, **kwargs):
        return 0

    def method_decorator(self, func):
        @functools.wraps(func)
        def wrapper(this, *args, **kwargs):
            self.before_call(*args, **kwargs)
            ret = func(this, *args, **kwargs)
            self.after_call(ret)
            return ret
        return wrapper

    def before_call(self, *args, **kwargs):
        pass

    def after_call(self, ret):
        pass
        