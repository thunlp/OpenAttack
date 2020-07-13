from ..attack_eval import AttackEval
from ..utils import check_parameters
from ..text_processors import DefaultTextProcessor
import numpy as np

DEFAULT_CONFIG = {
    "processor": DefaultTextProcessor(),
    "language_tool": None,
    "language_model": None,
    "sentence_encoder": None,

    "success_rate": True,       # 成功率
    "fluency": False,           # 流畅度
    "mistake": False,           # 语法错误
    "semantic": False,          # 语义匹配度
    "levenstein": False,        # 编辑距离
    "word_distance": False,     # 应用词级别编辑距离
    "modification_rate": False, # 修改率
}
class AttackEvalBase(AttackEval):
    def __init__(self, **kwargs):
        self.__config = DEFAULT_CONFIG.copy()
        self.__config.update(kwargs)
        check_parameters(DEFAULT_CONFIG.keys(), self.__config)
        self.clear()
    
    def __levenshtein(self, sentA, sentB):
        from ..metric import levenshtein
        return levenshtein(sentA, sentB)
    
    def __get_tokens(self, sent):
        return list(map(lambda x: x[0], self.__config["processor"].get_tokens(sent)))
    
    def __get_mistakes(self, sent):
        if self.__config["language_tool"] is None:
            import language_tool_python
            self.__config["language_tool"] = language_tool_python.LanguageTool('en-US')
        
        return len(self.__config["language_tool"].check(sent))
    
    def __get_fluency(self, sent):
        if self.__config["language_model"] is None:
            from ..metric import GPT2LM
            self.__config["language_model"] = GPT2LM()
        
        if len(sent.strip()) == 0:
            return 1
        return self.__config["language_model"](sent)
    
    def __get_semantic(self, sentA, sentB):
        if self.__config["sentence_encoder"] is None:
            from ..metric import UniversalSentenceEncoder
            self.__config["sentence_encoder"] = UniversalSentenceEncoder()
        
        return self.__config["sentence_encoder"](sentA, sentB)
    
    def __get_modification(self, sentA, sentB):
        from ..metric import modification
        tokenA = self.__get_tokens(sentA)
        tokenB = self.__get_tokens(sentB)
        return modification(tokenA, tokenB)
    
    def measure(self, input_, attack_result):
        if attack_result is None:
            return { "succeed": False }

        info = { "succeed": True }

        if self.__config["levenstein"]:
            va = input_
            vb = attack_result
            if self.__config["word_distance"]:
                va = self.__get_tokens(va)
                vb = self.__get_tokens(vb)
            info["edit"] =  self.__levenshtein(va, vb)
        
        if self.__config["mistake"]:
            info["mistake"] = self.__get_mistakes(attack_result)
        
        if self.__config["fluency"]:
            info["fluency"] = self.__get_fluency(attack_result)
            
        if self.__config["semantic"]:
            info["semantic"] = self.__get_semantic(input_, attack_result)

        if self.__config["modification_rate"]:
            info["modification"] = self.__get_modification(input_, attack_result)
        return info
        
    def update(self, info):
        if "total" not in self.__result:
            self.__result["total"] = 0
        self.__result["total"] += 1

        if self.__config["success_rate"]:
            if "succeed" not in self.__result:
                self.__result["succeed"] = 0
            if info["succeed"]:
                self.__result["succeed"] += 1
        
        # early stop
        if not info["succeed"]:
            return

        if self.__config["levenstein"]:
            if "edit" not in self.__result:
                self.__result["edit"] = 0
            self.__result["edit"] += info["edit"]
        
        if self.__config["mistake"]:
            if "mistake" not in self.__result:
                self.__result["mistake"] = 0
            self.__result["mistake"] += info["mistake"]

        if self.__config["fluency"]:
            if "fluency" not in self.__result:
                self.__result["fluency"] = 0
            self.__result["fluency"] += info["fluency"]

        if self.__config["semantic"]:
            if "semantic" not in self.__result:
                self.__result["semantic"] = 0
            self.__result["semantic"] += info["semantic"]
        
        if self.__config["modification_rate"]:
            if "modification" not in self.__result:
                self.__result["modification"] = 0
            self.__result["modification"] += info["modification"]
        return info
        
    def get_result(self):
        ret = {}
        ret["total"] = self.__result["total"]
        if self.__config["success_rate"]:
            ret["succeed"] = self.__result["succeed"]
            ret["success_rate"] = ret["succeed"] / ret["total"]
        if self.__result["succeed"] > 0:
            if self.__config["levenstein"]:
                if "edit" not in self.__result:
                    self.__result["edit"] = 0
                ret["levenstein"] = self.__result["edit"] / ret["succeed"]
            if self.__config["mistake"]:
                if "mistake" not in self.__result:
                    self.__result["mistake"] = 0
                ret["mistake"] = self.__result["mistake"] / ret["succeed"]
            if self.__config["fluency"]:
                if "fluency" not in self.__result:
                    self.__result["fluency"] = 0
                ret["fluency"] = self.__result["fluency"] / ret["succeed"]
            if self.__config["semantic"]:
                if "semantic" not in self.__result:
                    self.__result["semantic"] = 0
                ret["semantic"] = self.__result["semantic"] / ret["succeed"]
            if self.__config["modification_rate"]:
                if "modification" not in self.__result:
                    self.__result["modification"] = 0
                ret["modification_rate"] = self.__result["modification"] / ret["succeed"]
        return ret

    def clear(self):
        self.__result = {}
    
    def __del__(self):
        if self.__config["sentence_encoder"] is not None:
            del self.__config["sentence_encoder"]
        if self.__config["language_model"] is not None:
            del self.__config["language_model"]
        if self.__config["language_tool"] is not None:
            del self.__config["language_tool"]
