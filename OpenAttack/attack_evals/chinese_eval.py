from .default import DefaultAttackEval
from ..classifier import Classifier
from ..attacker import Attacker


class ChineseAttackEval(DefaultAttackEval):
    """
    ChineseAttackEval is the implementation of AttackEval for chinese datasets and models.
    """

    def __init__(self, attacker, classifier, **kwargs):
        """
        :param Attacker attacker: The attacker you use.
        :param Classifier classifier: The classifier you want to attack.
        :param kwargs: Other parameters, see :py:class:`.DefaultAttackEval` for detail.
        """
        super().__init__(attacker, classifier, **kwargs)

        self.__attacker = self.attacker
        self.__classifier = self.classifier

    def get_tokens(self, sent):
        import thulac
        tl = thulac.thulac(seg_only=True)
        return list(t[0] for t in tl.cut(sent))

    def get_fluency(self, sent):
        if self.__config["language_model"] is None:
            from ..metric import GPT2LMCH
            self.__config["language_model"] = GPT2LMCH()

        if len(sent.strip()) == 0:
            return 1
        return self.__config["language_model"](sent)

    def get_mistakes(self, sent):
        if self.__config["language_tool"] is None:
            import language_tool_python
            self.__config["language_tool"] = language_tool_python.LanguageTool('zh-CN')

        return len(self.__config["language_tool"].check(sent))
