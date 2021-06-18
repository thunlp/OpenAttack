from ...tags import *
class AttackMetric(object):
    """
    Base class of all metrics.
    """

    TAGS = { * TAG_ALL_LANGUAGE }

    def before_attack(self, input):
        return
    
    def after_attack(self, input, adversarial_sample):
        return
    
    @property
    def name(self):
        if hasattr(self, "NAME"):
            return self.NAME
        else:
            return self.__class__.__name__