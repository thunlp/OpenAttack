from ..attacker_eval import AttackerEval
from ..utils import check_parameters
from ..text_processors import DefaultTextProcessor
import numpy as np

DEFAULT_CONFIG = {
    "processor": DefaultTextProcessor(),
    "language_tool": None,
    "language_model": None,
    "sentence_encoder": None,

    "success_rate": True,   # 成功率
    "fluency": False,       # 流畅度
    "mistake": False,       # 语法错误
    "semantic": False,      # 语义匹配度
    "levenstein": False,    # 编辑距离
    "word_distance": False, # 应用词级别编辑距离
}
class AttackerEvalBase(AttackerEval):
    def __init__(self, **kwargs):
        self.__config = DEFAULT_CONFIG.copy()
        self.__config.update(kwargs)
        check_parameters(DEFAULT_CONFIG.keys(), self.__config)
        self.clear()
    
    def __levenshtein(self, a, b):
        la = len(a)
        lb = len(b)
        f = np.zeros((la + 1, lb + 1), dtype=np.uint64)
        for i in range(la + 1):
            for j in range(lb + 1):
                if i == 0:
                    f[i][j] = j
                elif j == 0:
                    f[i][j] = i
                elif a[i - 1] == b[j - 1]:
                    f[i][j] = f[i - 1][j - 1]
                else:
                    f[i][j] = min(f[i - 1][j - 1], f[i - 1][j], f[i][j - 1]) + 1
        return f[la][lb]
    
    def __get_tokens(self, sent):
        return list(map(lambda x: x[0], self.__config["processor"].get_tokens(sent)))
    
    def __get_mistakes(self, sent):
        if self.__config["language_tool"] is None:
            import language_tool_python
            self.__config["language_tool"] = language_tool_python.LanguageTool('en-US')
        
        return len(self.__config["language_tool"].check(sent))
    
    def __get_fluency(self, sent):
        if self.__config["language_model"] is None:
            from ..utils import GPT2LM
            self.__config["language_model"] = GPT2LM()
        
        return self.__config["language_model"](sent)
    
    def __get_semantic(self, sentA, sentB):
        if self.__config["sentence_encoder"] is None:
            from ..utils import UniversalSentenceEncoder
            self.__config["sentence_encoder"] = UniversalSentenceEncoder()
        
        return self.__config["sentence_encoder"](sentA, sentB)

    def update(self, input_, attack_result):
        if "total" not in self.__result:
            self.__result["total"] = 0
        self.__result["total"] += 1
        if self.__config["success_rate"]:
            if "succeed" not in self.__result:
                self.__result["succeed"] = 0
            if attack_result is not None:
                self.__result["succeed"] += 1
            
        if attack_result is None:
            return { "succeed": False }

        info = { "succeed": True }
        if self.__config["levenstein"]:
            va = input_
            vb = attack_result
            if self.__config["word_distance"]:
                va = self.__get_tokens(va)
                vb = self.__get_tokens(vb)
            rv = self.__levenshtein(va, vb)
            if "edit" not in self.__result:
                self.__result["edit"] = 0
            self.__result["edit"] += rv
            info["edit"] = rv
        
        if self.__config["mistake"]:
            if "mistake" not in self.__result:
                self.__result["mistake"] = 0
            rv = self.__get_mistakes(attack_result)
            self.__result["mistake"] += rv
            info["mistake"] = rv
        
        if self.__config["fluency"]:
            if "fluency" not in self.__result:
                self.__result["fluency"] = 0
            rv = self.__get_fluency(attack_result)
            self.__result["fluency"] += rv
            info["fluency"] += rv
        
        if self.__config["semantic"]:
            if "semantic" not in self.__result:
                self.__result["semantic"] = 0
            rv = self.__get_semantic(input_, attack_result)
            self.__result["semantic"] += rv
            info["semantic"] = rv
        return info
        
    def get_result(self):
        ret = {}
        ret["total"] = self.__result["total"]
        if self.__config["success_rate"]:
            ret["succeed"] = self.__result["succeed"]
            ret["success_rate"] = ret["succeed"] / ret["total"]
        if self.__config["levenstein"]:
            ret["levenstein"] = self.__result["edit"] / ret["succeed"]
        if self.__config["mistake"]:
            ret["mistake"] = self.__result["mistake"] / ret["succeed"]
        if self.__config["fluency"]:
            ret["fluency"] = self.__result["fluency"] / ret["succeed"]
        if self.__config["semantic"]:
            ret["semantic"] = self.__result["semantic"] / ret["succeed"]
        return ret

    def clear(self):
        self.__result = {}

