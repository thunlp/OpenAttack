import OpenAttack
import time
import multiprocessing


class TimeCalcClsf(OpenAttack.Classifier):
    def __init__(self, clsf):
        self.__clsf = clsf
        self.__total_time = multiprocessing.Value("d", 0.0)
    
    def get_pred(self, input_):
        st = time.perf_counter()
        ret = self.__clsf.get_pred(input_)
        ed = time.perf_counter()
        with self.__total_time.get_lock():
            self.__total_time.value += ed - st
        return ret
    
    def get_prob(self, input_):
        st = time.perf_counter()
        ret = self.__clsf.get_prob(input_)
        ed = time.perf_counter()
        with self.__total_time.get_lock():
            self.__total_time.value += ed - st
        return ret
    
    def get_grad(self, input_, labels):
        st = time.perf_counter()
        ret = self.__clsf.get_grad(input_, labels)
        ed = time.perf_counter()
        with self.__total_time.get_lock():
            self.__total_time.value += ed - st
        return ret

    @property
    def total_time(self):
        return self.__total_time.value
    
    def reset(self):
        with self.__total_time.get_lock():
            self.__total_time.value = 0