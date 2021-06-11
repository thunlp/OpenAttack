from typing import List, Optional
import datasets
from tqdm import tqdm

from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...text_process.tokenizer import Tokenizer, get_default_tokenizer
from ...utils import get_language, check_language, language_by_name
from ...tags import Tag

class UATAttacker(ClassificationAttacker):
    @property
    def TAGS(self):
        return { self.__lang_tag, Tag("get_pred", "victim") }

    def __init__(self,
            triggers : List[str] = ["the", "the", "the"],
            tokenizer : Optional[Tokenizer] = None,
            lang = None
        ):
        """
        :param list triggers: A list of trigger words.

        :Classifier Capacity: Gradient

        Universal Adversarial Triggers for Attacking and Analyzing NLP. Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, Sameer Singh. EMNLP-IJCNLP 2019. 
        `[pdf] <https://arxiv.org/pdf/1908.07125.pdf>`__
        `[code] <https://github.com/Eric-Wallace/universal-triggers>`__
        """
        lst = []
        if tokenizer is not None:
            lst.append(tokenizer)
        
        if len(lst) > 0:
            self.__lang_tag = get_language(lst)
        else:
            self.__lang_tag = language_by_name(lang)
            if self.__lang_tag is None:
                raise ValueError("Unknown language `%s`" % lang)

        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer

        check_language([self.tokenizer], self.__lang_tag)

        self.triggers = triggers

    
    def attack(self, victim: Classifier, sentence : str, goal : ClassifierGoal):
        trigger_sent = self.tokenizer.detokenize( self.triggers + self.tokenizer.tokenize(sentence, pos_tagging=False) )
        pred = victim.get_pred([trigger_sent])[0]
        if goal.check(trigger_sent, pred):
            return trigger_sent
        return None

        
    @classmethod
    def get_triggers(self,
            victim : Classifier, 
            dataset : datasets.Dataset, 
            tokenizer : Tokenizer,
            epoch : int = 5,
            batch_size : int = 5,
            trigger_len : int = 3,
            beam_size : int = 5,
            lang = None
        ):
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

        requires = [ Tag("get_grad", "victim"), Tag("get_prob", "victim"), Tag("get_embedding", "victim"), Tag("get_pred", "victim") ]
        for tag in requires:
            if tag not in victim.TAGS:
                raise AttributeError("`%s` requires victim to support `%s`" % (self.__class__.__name__, tag.name))
        
        if tokenizer is not None:
            lang_tag = language_by_name(lang)
            if lang_tag is None:
                raise ValueError("Invalid language type `%s`" % lang)
            tokenizer = get_default_tokenizer(lang_tag)


        victim_embedding = victim.get_embedding()

        word2id = victim_embedding.word2id
        embedding = victim_embedding.embedding

        id2word = { v: k for k, v in word2id.items() }

        def get_candidates(gradient):
            idx = embedding.dot( gradient.T ).argsort()[:beam_size].tolist()
            return [ id2word[id_] for id_ in idx]
        
        curr_trigger = ["the" for _ in range(trigger_len)]
        for epoch_idx in range(epoch):
            for num_iter in tqdm( range( (len(dataset) + batch_size - 1) // batch_size ), desc="Epoch %d: " % epoch_idx ):
                cnt = num_iter * batch_size
                batch = dataset[ cnt: cnt + batch_size ]

                x = [
                    tokenizer.tokenize(sent, pos_tagging=False)
                        for sent in batch["x"]
                ]
                y = batch["y"]

                nw_beams = [ ( curr_trigger,  0 ) ]
                for i in range(trigger_len):
                    # beam search here
                    beams = nw_beams
                    nw_beams = []
                    for trigger, _ in beams:
                        xt = list(map(lambda x: trigger + x, x))
                        grad = victim.get_grad(xt, y)[1]
                        candidates_words = get_candidates(grad[:, i, :].mean(axis=0))

                        for cw in candidates_words:
                            tt = trigger[:i] + [cw] + trigger[i + 1:]
                            xt = list(map(lambda x:  tokenizer.detokenize(tt + x), x))
                            pred = victim.get_prob(xt)
                            loss = pred[ (list(range(len(y))), list(y) ) ].sum()
                            nw_beams.append((tt, loss))
                    nw_beams = sorted(nw_beams, key=lambda x: x[1])[:beam_size]
                curr_trigger = nw_beams[0][0]
        return curr_trigger
                        
