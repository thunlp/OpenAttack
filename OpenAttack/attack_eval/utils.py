from ..victim import Victim
from ..exceptions import InvokeLimitExceeded
import logging

logger = logging.getLogger("OpenAttack.AttackEval")

def attack_process(attacker, victim : Victim, data, limit):
    victim.set_context(data, limit)
    try:
        adversarial_sample = attacker(victim, data)
        invoke_times = victim.context.invoke
        attack_time = victim.context.attack_time
    except InvokeLimitExceeded:
        adversarial_sample = None
        invoke_times = victim.context.invoke + 1
        attack_time = victim.context.attack_time
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        logger.exception("Exception when evaluate data %s", data)
        adversarial_sample = None
        invoke_times = victim.context.invoke
        attack_time = victim.context.attack_time
    finally:
        victim.clear_context()
    
    return adversarial_sample, attack_time, invoke_times
    

def worker_process(data):
    attacker = globals()["$WORKER_ATTACKER"]
    victim = globals()["$WORKER_VICTIM"]
    limit = globals()["$WORKER_INVOKE_LIMIT"]

    return attack_process(attacker, victim, data, limit)



def worker_init(attacker, victim, limit):
    globals()['$WORKER_ATTACKER'] = attacker
    globals()['$WORKER_VICTIM'] = victim
    globals()['$WORKER_INVOKE_LIMIT'] = limit
