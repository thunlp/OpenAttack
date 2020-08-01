import numpy as np
import os
from ..text_processors import DefaultTextProcessor
from ..substitutes import CounterFittedSubstitute
from ..exceptions import WordNotInDictionaryException
from ..utils import check_parameters
from ..metric import usencoder
from ..attacker import Attacker

DEFAULT_SKIP_WORDS = set(
    [
        "the",
        "and",
        "a",
        "of",
        "to",
        "is",
        "it",
        "in",
        "i",
        "this",
        "that",
        "was",
        "as",
        "for",
        "with",
        "movie",
        "but",
        "film",
        "on",
        "not",
        "you",
        "he",
        "are",
        "his",
        "have",
        "be",
    ]
)

DEFAULT_CONFIG = {
    "skip_words": DEFAULT_SKIP_WORDS,
    "import_score_threshold": -1.,
    "sim_score_threshold": 0.5, 
    "sim_score_window": 15, 
    "synonym_num": 50,
    "processor": DefaultTextProcessor(),
    "substitute": None,
}


class TextFoolerAttacker(Attacker):
    def __init__(self, **kwargs):
        """
        :param list skip_words: A list of words which won't be replaced during the attack. **Default:** A list of words that is most frequently used.
        :param float import_score_threshold: Threshold used to choose important word. **Default:** -1.
        :param float sim_score_threshold: Threshold used to choose sentences of high semantic similarity. **Default:** 0.5
        :param int sim_score_window: length used in score module. **Default:** 15
        :param int synonym_num: Maximum candidates of word substitution. **Default:** 50
        :param TextProcessor processor: Text processor used in this attacker. **Default:** :any:`DefaultTextProcessor`
        :param WordSubstitute substitute: Substitute method used in this attacker. **Default:** :any:`CounterFittedSubstitute()`

        :Classifier Capacity: Score

        Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment. Di Jin, Zhijing Jin, Joey Tianyi Zhou, Peter Szolovits. AAAI 2020.
        `[pdf] <https://arxiv.org/pdf/1907.11932v4>`__
        `[code] <https://github.com/jind11/TextFooler>`__
        
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        if self.config["substitute"] is None:
            self.config["substitute"] = CounterFittedSubstitute(cosine=True)

        check_parameters(DEFAULT_CONFIG.keys(), self.config)
        self.sim_predictor = usencoder.UniversalSentenceEncoder()

    def __call__(self, clsf, x_orig, target=None):
        """
        * **clsf** : **Classifier** .
        * **x_orig** : Input sentence.
        """
        x_orig = x_orig.lower()
        x_copy = x_orig
        if target is None:
            targeted = False
            target = clsf.get_pred([x_orig])[0]  
        else:
            targeted = True
        orig_probs = clsf.get_prob([x_orig])
        orig_label = clsf.get_pred([x_orig])
        orig_prob = orig_probs.max()

        x_orig = self.config["processor"].get_tokens(x_orig)
        x_pos =  list(map(lambda x: x[1], x_orig))
        x_orig = list(map(lambda x: x[0], x_orig))

        len_text = len(x_orig)
        if len_text < self.config["sim_score_window"]:
            self.config["sim_score_threshold"] = 0.1  
        half_sim_score_window = (self.config["sim_score_window"] - 1) // 2



        pos_ls = list(map(lambda x: x[1], self.config["processor"].get_tokens(x_copy)))



        leave_1_texts = [x_orig[:ii] + ['<oov>'] + x_orig[min(ii + 1, len_text):] for ii in range(len_text)]
        leave_1_probs = clsf.get_prob([self.config["processor"].detokenizer(sentence) for sentence in leave_1_texts])
        leave_1_probs_argmax = np.argmax(leave_1_probs, axis=-1)

        import_scores = orig_prob - leave_1_probs[:, orig_label].squeeze() + (leave_1_probs_argmax != orig_label).astype(np.float64) * (
                    np.max(leave_1_probs, axis=-1) - orig_probs.squeeze()[leave_1_probs_argmax])

        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > self.config["import_score_threshold"] and x_orig[idx] not in self.config["skip_words"]:
                    words_perturb.append((idx, x_orig[idx], x_pos[idx]))
            except:
                print(idx, len(x_orig), import_scores.shape, x_orig, len(leave_1_texts))



        synonym_words = [
            self.get_neighbours(word, pos)
            if word not in self.config["skip_words"]
            else []
            for idx, word, pos in words_perturb
        ]
        synonyms_all = []
        for idx, word, pos in words_perturb:
            synonyms = synonym_words.pop(0)
            if synonyms:
                synonyms_all.append((idx, synonyms))


        text_prime = x_orig[:]
        text_cache = text_prime[:]
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = clsf.get_prob([self.config["processor"].detokenizer(sentence) for sentence in new_texts])


            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = self.config["sim_score_window"]
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - self.config["sim_score_window"]
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text

            texts = [self.config["processor"].detokenizer(x[text_range_min:text_range_max]) for x in new_texts]
            semantic_sims = np.array([self.sim_predictor(self.config["processor"].detokenizer(text_cache[text_range_min:text_range_max]), x) for x in texts])
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = orig_label != np.argmax(new_probs, axis=-1)

            new_probs_mask *= (semantic_sims >= self.config["sim_score_threshold"])

            synonyms_pos_ls = [list(map(lambda x: x[1], self.config["processor"].get_tokens(self.config["processor"].detokenizer(new_text[max(idx - 4, 0):idx + 5]))))[min(4, idx)]
                               if len(new_text) > 10 else list(map(lambda x: x[1], self.config["processor"].get_tokens(self.config["processor"].detokenizer(new_text))))[idx] for new_text in new_texts]

            pos_mask = np.array(self.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                x_adv = self.config["processor"].detokenizer(text_prime)
                pred = clsf.get_pred([x_adv])
                if not targeted:
                    return (x_adv, pred[0])
                elif pred[0] == target:
                    return (x_adv, pred[0])
            else:
                new_label_probs = new_probs[:, orig_label] + (semantic_sims < self.config["sim_score_threshold"]) + (1 - pos_mask).astype(np.float64)
                
                new_label_prob_min = np.min(new_label_probs, axis=0)[0]
                new_label_prob_argmin = np.argmin(new_label_probs, axis=0)[0]
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
            text_cache = text_prime[:]
        return None
            

    def get_neighbours(self, word, pos):
        threshold = 0.5
        try:
            return list(
                map(
                    lambda x: x[0],
                    self.config["substitute"](word, pos=pos, threshold=threshold)[1 : self.config["synonym_num"] + 1],
                )
            )
        except WordNotInDictionaryException:
            return []


    def pos_filter(self, ori_pos, new_pos_list):
        same = [True if ori_pos == new_pos or (set([ori_pos, new_pos]) <= set(['NOUN', 'VERB']))
                else False for new_pos in new_pos_list]
        return same