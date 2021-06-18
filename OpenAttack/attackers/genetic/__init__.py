from typing import List, Optional
import numpy as np
from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...text_process.tokenizer import get_default_tokenizer, Tokenizer
from ...attack_assist.substitute.word import WordSubstitute, get_default_substitute
from ...utils import get_language, check_language, language_by_name
from ...exceptions import WordNotInDictionaryException
from ...attack_assist.filter_words import get_default_filter_words
from ...tags import Tag

class GeneticAttacker(ClassificationAttacker):

    @property
    def TAGS(self):
        return { self.__lang_tag, Tag("get_pred", "victim"), Tag("get_prob", "victim") }

    def __init__(self, 
            pop_size : int = 20, 
            max_iters : int = 20, 
            tokenizer : Optional[Tokenizer] = None, 
            substitute : Optional[WordSubstitute] = None, 
            lang = None,
            filter_words : List[str] = None
        ):
        """
        Generating Natural Language Adversarial Examples. Moustafa Alzantot, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani Srivastava, Kai-Wei Chang. EMNLP 2018.
        `[pdf] <https://www.aclweb.org/anthology/D18-1316.pdf>`__
        `[code] <https://github.com/nesl/nlp_adversarial_examples>`__
        
        Args:
            pop_size: Genetic algorithm popluation size. **Default:** 20
            max_iter: Maximum generations of genetic algorithm. **Default:** 20
            tokenizer: A tokenizer that will be used during the attack procedure. Must be an instance of :py:class:`.Tokenizer`
            substitute: A substitute that will be used during the attack procedure. Must be an instance of :py:class:`.WordSubstitute`
            lang: The language used in attacker. If is `None` then `attacker` will intelligently select the language based on other parameters.
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
        
        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer
        
        if substitute is None:
            substitute = get_default_substitute(self.__lang_tag)
        self.substitute = substitute
        self.pop_size = pop_size
        self.max_iters = max_iters

        if filter_words is None:
            filter_words = get_default_filter_words(self.__lang_tag)
        self.filter_words = set(filter_words)

        check_language([self.tokenizer, self.substitute], self.__lang_tag)


    def attack(self, victim: Classifier, x_orig, goal: ClassifierGoal):
        x_orig = x_orig.lower()
        
        x_orig = self.tokenizer.tokenize(x_orig)
        x_pos =  list(map(lambda x: x[1], x_orig))
        x_orig = list(map(lambda x: x[0], x_orig))

        neighbours_nums = [
            self.get_neighbour_num(word, pos) if word not in self.filter_words else 0
            for word, pos in zip(x_orig, x_pos)
        ]
        neighbours = [
            self.get_neighbours(word, pos)
            if word not in self.filter_words
            else []
            for word, pos in zip(x_orig, x_pos)
        ]

        if np.sum(neighbours_nums) == 0:
            return None
        w_select_probs = neighbours_nums / np.sum(neighbours_nums)

        pop = [  # generate population
            self.perturb(
                victim, x_orig, x_orig, neighbours, w_select_probs, goal
            )
            for _ in range(self.pop_size)
        ]
        for i in range(self.max_iters):
            pop_preds = victim.get_prob(self.make_batch(pop))

            if goal.targeted:
                top_attack = np.argmax(pop_preds[:, goal.target])
                if np.argmax(pop_preds[top_attack, :]) == goal.target:
                    return self.tokenizer.detokenize(pop[top_attack])
            else:
                top_attack = np.argmax(-pop_preds[:, goal.target])
                if np.argmax(pop_preds[top_attack, :]) != goal.target:
                    return self.tokenizer.detokenize(pop[top_attack])

            pop_scores = pop_preds[:, goal.target]
            if not goal.targeted:
                pop_scores = 1.0 - pop_scores

            if np.sum(pop_scores) == 0:
                return None
            pop_scores = pop_scores / np.sum(pop_scores)

            elite = [pop[top_attack]]
            parent_indx_1 = np.random.choice(
                self.pop_size, size=self.pop_size - 1, p=pop_scores
            )
            parent_indx_2 = np.random.choice(
                self.pop_size, size=self.pop_size - 1, p=pop_scores
            )
            childs = [
                self.crossover(pop[p1], pop[p2])
                for p1, p2 in zip(parent_indx_1, parent_indx_2)
            ]
            childs = [
                self.perturb(
                    victim, x_cur, x_orig, neighbours, w_select_probs, goal
                )
                for x_cur in childs
            ]
            pop = elite + childs

        return None  # Failed

    def get_neighbour_num(self, word, pos):
        try:
            return len(self.substitute(word, pos))
        except WordNotInDictionaryException:
            return 0

    def get_neighbours(self, word, pos):
        try:
            return list(
                map(
                    lambda x: x[0],
                    self.substitute(word, pos),
                )
            )
        except WordNotInDictionaryException:
            return []

    def select_best_replacements(
        self, clsf, indx, neighbours, x_cur, x_orig, goal : ClassifierGoal
    ):
        def do_replace(word):
            ret = x_cur.copy()
            ret[indx] = word
            return ret
        new_list = []
        rep_words = []
        for word in neighbours:
            if word != x_orig[indx]:
                new_list.append(do_replace(word))
                rep_words.append(word)
        if len(new_list) == 0:
            return x_cur
        new_list.append(x_cur)

        pred_scores = clsf.get_prob(self.make_batch(new_list))[:, goal.target]
        if goal.targeted:
            new_scores = pred_scores[:-1] - pred_scores[-1]
        else:
            new_scores = pred_scores[-1] - pred_scores[:-1]

        if np.max(new_scores) > 0:
            return new_list[np.argmax(new_scores)]
        else:
            return x_cur

    def make_batch(self, sents):
        return [self.tokenizer.detokenize(sent) for sent in sents]

    def perturb(
        self, clsf, x_cur, x_orig, neighbours, w_select_probs, goal : ClassifierGoal
    ):
        x_len = len(x_cur)
        num_mods = 0
        for i in range(x_len):
            if x_cur[i] != x_orig[i]:
                num_mods += 1
        mod_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        if num_mods < np.sum(
            np.sign(w_select_probs)
        ):  # exists at least one indx not modified
            while x_cur[mod_idx] != x_orig[mod_idx]:  # already modified
                mod_idx = np.random.choice(x_len, 1, p=w_select_probs)[
                    0
                ]  # random another indx
        return self.select_best_replacements(
            clsf, mod_idx, neighbours[mod_idx], x_cur, x_orig, goal
        )

    def crossover(self, x1, x2):
        ret = []
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                ret.append(x1[i])
            else:
                ret.append(x2[i])
        return ret
