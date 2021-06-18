class AttackGoal:
    def check(self, adversarial_sample, prediction) -> bool:
        raise NotImplementedError()