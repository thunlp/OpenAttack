from .base import MetricSelector
class GrammaticalErrors(MetricSelector):
    def _select(self, lang):
        if lang.name == "english":
            from ..algorithms.language_tool import LanguageTool
            return LanguageTool()
        if lang.name == "chinese":
            from ..algorithms.language_tool import LanguageToolChinese
            return LanguageToolChinese()
    