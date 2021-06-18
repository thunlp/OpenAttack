from typing import List, Optional
import numpy as np

from ...metric import UniversalSentenceEncoder
from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...text_process.tokenizer import Tokenizer, get_default_tokenizer
from ...attack_assist.substitute.word import WordSubstitute, get_default_substitute
from ...utils import get_language, check_language, language_by_name
from ...exceptions import WordNotInDictionaryException
from ...tags import Tag
from ...attack_assist.filter_words import get_default_filter_words

class TextFoolerAttacker(ClassificationAttacker):
    @property
    def TAGS(self):
        return { self.__lang_tag, Tag("get_pred", "victim"), Tag("get_prob", "victim") }

    def __init__(self,
            import_score_threshold : float = -1,
            sim_score_threshold : float = 0.5,
            sim_score_window : int = 15,
            tokenizer : Optional[Tokenizer] = None,
            substitute : Optional[WordSubstitute] = None,
            filter_words : List[str] = None,
            token_unk = "<UNK>",
            lang = None,
        ):
        """
        Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment. Di Jin, Zhijing Jin, Joey Tianyi Zhou, Peter Szolovits. AAAI 2020.
        `[pdf] <https://arxiv.org/pdf/1907.11932v4>`__
        `[code] <https://github.com/jind11/TextFooler>`__

        Args:
            import_score_threshold: Threshold used to choose important word. **Default:** -1.
            sim_score_threshold: Threshold used to choose sentences of high semantic similarity. **Default:** 0.5
            im_score_window: length used in score module. **Default:** 15
            tokenizer: A tokenizer that will be used during the attack procedure. Must be an instance of :py:class:`.Tokenizer`
            substitute: A substitute that will be used during the attack procedure. Must be an instance of :py:class:`.WordSubstitute`
            lang: The language used in attacker. If is `None` then `attacker` will intelligently select the language based on other parameters.
            token_unk: The token id or the token name for out-of-vocabulary words in victim model. **Default:** ``"<UNK>"``
            filter_words: A list of words that will be preserved in the attack procesudre.

        :Classifier Capacity:
            * get_pred
            * get_prob
        
        """
        lst = []
        if tokenizer is not None:
            lst.append(tokenizer)
        if substitute is not None:
            lst.append(substitute)
        if len(lst) > 0:
            self.__lang_tag = get_language(lst)
        else:
            self.__lang_tag = language_by_name(lang)
            if self.__lang_tag is None:
                raise ValueError("Unknown language `%s`" % lang)
        
        if substitute is None:
            substitute = get_default_substitute(self.__lang_tag)
        self.substitute = substitute

        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer

        self.sim_predictor = UniversalSentenceEncoder()

        check_language([self.tokenizer, self.substitute, self.sim_predictor], self.__lang_tag)

        self.import_score_threshold = import_score_threshold
        self.sim_score_threshold = sim_score_threshold
        self.sim_score_window = sim_score_window

        if filter_words is None:
            filter_words = get_default_filter_words(self.__lang_tag)
        self.filter_words = set(filter_words)

        self.token_unk = token_unk

    def attack(self, victim: Classifier, sentence : str, goal: ClassifierGoal):
        """
        * **clsf** : **Classifier** .
        * **x_orig** : Input sentence.
        """
        x_orig = sentence.lower()

        orig_probs = victim.get_prob([x_orig])[0]
        orig_label = orig_probs.argmax()
        orig_prob = orig_probs.max()

        x_orig = self.tokenizer.tokenize(x_orig)
        x_pos =  list(map(lambda x: x[1], x_orig))
        x_orig = list(map(lambda x: x[0], x_orig))

        len_text = len(x_orig)
        if len_text < self.sim_score_window:
            self.sim_score_threshold = 0.1  
        half_sim_score_window = (self.sim_score_window - 1) // 2


        leave_1_texts = [x_orig[:ii] + [self.token_unk] + x_orig[min(ii + 1, len_text):] for ii in range(len_text)]
        leave_1_probs = victim.get_prob([self.tokenizer.detokenize(sentence) for sentence in leave_1_texts])
        leave_1_probs_argmax = np.argmax(leave_1_probs, axis=1)

        import_scores = orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).astype(np.float64) * (
                    np.max(leave_1_probs, axis=1) - orig_probs[leave_1_probs_argmax])

        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            if score > self.import_score_threshold and x_orig[idx] not in self.filter_words:
                words_perturb.append((idx, x_orig[idx], x_pos[idx]))

        synonym_words = [
            self.get_neighbours(word, pos)
            if word not in self.filter_words
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
            new_texts = [text_prime[:idx] + [synonym] + text_prime[idx + 1:] for synonym in synonyms]
            new_probs = victim.get_prob([self.tokenizer.detokenize(sentence) for sentence in new_texts])


            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = self.sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - self.sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text

            texts = [self.tokenizer.detokenize(x[text_range_min:text_range_max]) for x in new_texts]
            semantic_sims = np.array([self.sim_predictor.calc_score(self.tokenizer.detokenize(text_cache[text_range_min:text_range_max]), x) for x in texts])
            new_probs_mask = orig_label != np.argmax(new_probs, axis=1)

            new_probs_mask *= (semantic_sims >= self.sim_score_threshold)

            synonyms_pos_ls = [list(map(lambda x: x[1], self.tokenizer.tokenize(self.tokenizer.detokenize(new_text[max(idx - 4, 0):idx + 5]))))[min(4, idx)]
                               if len(new_text) > 10 else list(map(lambda x: x[1], self.tokenizer.tokenize(self.tokenizer.detokenize(new_text))))[idx] for new_text in new_texts]

            pos_mask = np.array(self.pos_filter(x_pos[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                x_adv = self.tokenizer.detokenize(text_prime)
                pred = victim.get_pred([x_adv])[0]
                if goal.check(x_adv, pred):
                    return x_adv
            else:
                new_label_probs = new_probs[:, orig_label] + (semantic_sims < self.sim_score_threshold).astype(np.float64) + (1 - pos_mask).astype(np.float64)
                
                new_label_prob_min = np.min(new_label_probs, axis=0)
                new_label_prob_argmin = np.argmin(new_label_probs, axis=0)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
            text_cache = text_prime[:]
        return None
            

    def get_neighbours(self, word, pos):
        try:
            return list(
                filter(
                    lambda x: x != word,
                    map(
                        lambda x: x[0],
                        self.substitute(word, pos),
                    )
                )
            )
        except WordNotInDictionaryException:
            return []


    def pos_filter(self, ori_pos, new_pos_list):
        same = [True if ori_pos == new_pos or (set([ori_pos, new_pos]) <= set(['noun', 'verb']))
                else False for new_pos in new_pos_list]
        return same