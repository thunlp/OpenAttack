from ...tags import Tag
from ..algorithms.base import AttackMetric
class MetricSelector:
    def select(self, lang : Tag) -> AttackMetric:
        return self._select(lang)
    
    def _select(self, lang):
        raise NotImplementedError()