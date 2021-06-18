from .base import MetricSelector
from ...text_process.tokenizer import get_default_tokenizer
class ModificationRate(MetricSelector):
    """
    :English: `Modification` ( :py:class:`.PunctTokenizer` )
    :Chinese: `Modification` ( :py:class:`.JiebaTokenizer` )
    """

    def _select(self, lang):
        from ..algorithms.modification import Modification
        return Modification( get_default_tokenizer(lang) )