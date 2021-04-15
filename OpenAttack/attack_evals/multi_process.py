from .default import DefaultAttackEval
import multiprocessing, logging

logger = logging.getLogger(__name__)

# TODO: not tested on tf yet

def worker(data):
    
    attacker = globals()["$WORKER_ATTACKER"]
    classifier = globals()["$WORKER_CLASSIFIER"]
    if "target" in data:
        res = attacker(classifier, data["x"], data["target"])
    else:
        res = attacker(classifier, data["x"])
    return data, res

def worker_init(attacker, classifier):
    globals()['$WORKER_ATTACKER'] = attacker
    globals()['$WORKER_CLASSIFIER'] = classifier

class MultiProcessEvalMixin(object):
    def __init__(self, *args, num_process=1, chunk_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_process = num_process
        if chunk_size is None:
            self.chunk_size = self.num_process
    
    def __update(self, sentA, sentB):
        info = self.measure(sentA, sentB)
        return self.update(info)

    def eval_results(self, dataset):
        self.clear()

        def _iter_gen():
            for data in dataset:
                yield data
        
        if multiprocessing.get_start_method() != "spawn":
            logger.warning("Warning: multiprocessing start method '%s' may cause pytorch.cuda initialization error.", multiprocessing.get_start_method())
        
        with multiprocessing.Pool(self.num_process, initializer=worker_init, initargs=(self.attacker, self.classifier)) as pool:
            for data, res in pool.imap(worker, _iter_gen(), chunksize=self.chunk_size):
                if res is None:
                    info = self.__update(data["x"], None)
                else:
                    info = self.__update(data["x"], res[0])
                if not info["Succeed"]:
                    yield (data, None, None, info)
                else:
                    yield (data, res[0], res[1], info)

class MultiProcessAttackEval(MultiProcessEvalMixin, DefaultAttackEval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)