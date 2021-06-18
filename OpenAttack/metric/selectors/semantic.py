from .base import MetricSelector
class SemanticSimilarity(MetricSelector):
    """
    :English: :py:class:`.UniversalSentenceEncoder`
    """

    def _select(self, lang):
        if lang.name == "english":
            from ..algorithms.usencoder import UniversalSentenceEncoder
            return UniversalSentenceEncoder()
        