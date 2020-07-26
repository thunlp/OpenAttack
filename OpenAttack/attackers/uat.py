import numpy as np
from ..text_processors import DefaultTextProcessor
from ..utils import check_parameters
from ..attacker import Attacker
from ..exceptions import WordNotInDictionaryException, NoEmbeddingException
from tqdm import tqdm

DEFAULT_CONFIG = {
    "triggers": ["the", "the", "the"],
    "processor": DefaultTextProcessor()
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
        """
        :param list triggers: A list of trigger words.

        :Classifier Capacity: Gradient

        Universal Adversarial Triggers for Attacking and Analyzing NLP. Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, Sameer Singh. EMNLP-IJCNLP 2019. 
        `[pdf] <https://arxiv.org/pdf/1908.07125.pdf>`__
        `[code] <https://github.com/Eric-Wallace/universal-triggers>`__
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        check_parameters(DEFAULT_CONFIG.keys(), self.config)
    
    def __call__(self, clsf, x_orig, target=None):
        if target is None:
            targeted = False
            target = clsf.get_pred([x_orig])[0]  # calc x_orig's prediction
        else:
            targeted = True
        trigger_sent = self.config["processor"].detokenizer(self.config["triggers"]) + " " + x_orig
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
        """
        :param Classifier clsf: The classifier that you want to attack.
        :param Dataset dataset: A :py:class:`.Dataset` or a list of :py:class:`.DataInstance`.
        :param np.ndarray embedding: The 2d word vector matrix of shape (vocab_size, vector_dim).
        :param dict word2id: A dict that maps tokens to ids.
        :param int epoch: Maximum epochs to get the universal adversarial triggers.
        :param int barch_size: Batch size.
        :param int trigger_len: The number of triggers.
        :param int beam_size: Beam search size used in UATAttacker.
        :return: Returns a list of triggers which can be directly used in ``__init__``
        :rtype: list
        """
        config = TRAIN_CONFIG.copy()
        config.update(kwargs)
        check_parameters(TRAIN_CONFIG.keys(), config)

        if (config["word2id"] is None) or (config["embedding"] is None):
            raise NoEmbeddingException()

        id2word = { v: k for k, v in config["word2id"].items() }

        def get_candidates(gradient):
            idx = config["embedding"].dot( gradient.T ).argsort()[:config["beam_size"]].tolist()
            return [ id2word[id_] for id_ in idx]
        
        curr_trigger = ["the" for _ in range(config["trigger_len"])]
        for epoch in range(config["epoch"]):
            for num_iter in tqdm( range( (len(dataset) + config["batch_size"] - 1) // config["batch_size"] ) ):
                cnt = num_iter * config["batch_size"]
                batch = dataset[ cnt: cnt + config["batch_size"] ]
                cnt += config["batch_size"]

                x = [
                    list(map(lambda x: x[0], config["processor"].get_tokens(sent)))
                        for sent in list(map(lambda x: x.x, batch))
                ]
                y = list(map(lambda x: x.y, batch))

                nw_beams = [ ( curr_trigger,  0 ) ]
                for i in range(config["trigger_len"]):
                    # beam search here
                    beams = nw_beams
                    nw_beams = []
                    for trigger, _ in beams:
                        xt = list(map(lambda x: trigger + x, x))
                        grad = clsf.get_grad(xt, y)[1]
                        candidates_words = get_candidates(grad[:, i, :].mean(axis=0))

                        for cw in candidates_words:
                            tt = trigger[:i] + [cw] + trigger[i + 1:]
                            xt = list(map(lambda x:  config["processor"].detokenizer(tt + x), x))
                            pred = clsf.get_prob(xt)
                            loss = pred[ (list(range(len(y))), list(y) ) ].sum()
                            nw_beams.append((tt, loss))
                    nw_beams = sorted(nw_beams, key=lambda x: x[1])[:config["beam_size"]]
                curr_trigger = nw_beams[0][0]
        return curr_trigger
                        
