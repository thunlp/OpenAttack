from ...tags import Tag
from ..algorithms.base import AttackMetric
class MetricSelector:
    """
    Base class of all metric selectors.

    MetricSelector is a helper class for OpenAttack to select AttackMetric by language.
    """
    def select(self, lang : Tag) -> AttackMetric:
        return self._select(lang)
    
    def _select(self, lang):
        raise NotImplementedError()