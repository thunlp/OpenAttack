import numpy as np
from ..text_processors import DefaultTextProcessor
from ..substitutes import CounterFittedSubstitute   # TODO: replace it to WordNet !!
from ..utils import check_parameters
from ..attacker import Attacker
from ..exceptions import WordNotInDictionaryException, NoEmbeddingException

DEFAULT_CONFIG = {
    "triggers": []
}

TRAIN_CONFIG = {
    "processor": DefaultTextProcessor(),
    "embedding": None,
    "word2id": None,
    "epoch": 5,
    "batch_size": 32,
    "trigger_len": 3,
    "beam_size": 5,
}

class UATAttacker(Attacker):
    def __init__(self, **kwargs):
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        check_parameters(self.config.keys(), DEFAULT_CONFIG)
    
    def __call__(self, clsf, x_orig, target=None):
        if target is None:
            targeted = False
            target = clsf.get_pred([x_orig])[0]  # calc x_orig's prediction
        else:
            targeted = True
        trigger_sent = " ".join(self.config["triggers"]) + " " + x_orig
        pred = clsf.get_pred([trigger_sent])[0]
        if pred == target:
            if targeted:
                return ( trigger_sent, pred )
            else:
                return None
        else:
            if targeted:
                return None
            else:
                return ( trigger_sent, pred )

    @classmethod
    def get_triggers(self, clsf, dataset, **kwargs):
        config = TRAIN_CONFIG.copy()
        config.update(kwargs)
        check_parameters(config.keys(), DEFAULT_CONFIG)

        if (config["word2id"] is None) or (config["embedding"] is None):
            raise NoEmbeddingException()

        id2word = { v: k for k, v in config["word2id"].items() }

        def get_candidates(gradient):
            idx = config["embedding"].dot( gradient.T ).argsort()[:config["beam_size"]].tolist()
            return [ id2word[id_] for id_ in idx]
        
        curr_trigger = ["the" for _ in range(config["trigger_len"])]
        for epoch in range(config["epoch"]):
            cnt = 0
            while cnt < len(dataset):
                batch = dataset[ cnt: cnt + config["batch_size"] ]
                cnt += config["batch_size"]

                x = list(map(lambda x: x[0].lower(), batch))
                y = list(map(lambda x: x[1], batch))

                nw_beams = [ ( curr_trigger,  0 ) ]
                for i in range(config["trigger_len"]):
                    # beam search here
                    beams = nw_beams
                    nw_beams = []
                    for trigger, _ in beams:
                        trigger_sent = " ".join(trigger) + " "
                        xt = list(map(lambda x: trigger_sent + x, x))
                        grad = clsf.get_grad(xt, y)[1]
                        candidates_words = get_candidates(grad[:, i, :].mean(axis=0))

                        for cw in candidates_words:
                            tt = trigger[:i] + [cw] + trigger[i + 1:]
                            trigger_sent = " ".join(tt) + " "
                            xt = list(map(lambda x: trigger_sent + x, x))
                            pred = clsf.get_prob(xt)
                            loss = 0
                            for j in range(pred.shape[0]):
                                loss += pred[j, y[j]]
                            nw_beams.append((tt, loss))
                    nw_beams = sorted(nw_beams, key=lambda x: x[1])[:config["beam_size"]]
                curr_trigger = nw_beams[0][0]
        return curr_trigger
                        






        



