from .base import MetricSelector
class SemanticSimilarity(MetricSelector):
    def _select(self, lang):
        if lang == "english":
            from ..algorithms.usencoder import UniversalSentenceEncoder
            return UniversalSentenceEncoder()
        