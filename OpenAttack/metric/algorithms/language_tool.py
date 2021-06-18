from .base import AttackMetric
from ...tags import *

class LanguageTool(AttackMetric):
    
    
    NAME = "Grammatical Errors"
    TAGS = { TAG_English }

    def __init__(self) -> None:
        """
        Use language_tool_python to check grammer.

        :Package Requirements:
            * language_tool_python
        :Language: english

        """
        import language_tool_python
        self.language_tool = language_tool_python.LanguageTool('en-US')
    
    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return len(self.language_tool.check(adversarial_sample))

class LanguageToolChinese(AttackMetric):
    
    NAME = "Grammatical Errors"
    TAGS = { TAG_Chinese }

    def __init__(self) -> None:
        """
        Use language_tool_python to check grammer.

        :Package Requirements:
            * language_tool_python
        :Language: chinese

        """
        import language_tool_python
        self.language_tool = language_tool_python.LanguageTool('zh-CN')
    
    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return len(self.language_tool.check(adversarial_sample))
    