from .base import MetricSelector
from ...text_process.tokenizer import get_default_tokenizer

class EditDistance(MetricSelector):
    def _select(self, lang):
        from ..algorithms.levenshtein import Levenshtein
        return Levenshtein( get_default_tokenizer(lang) )