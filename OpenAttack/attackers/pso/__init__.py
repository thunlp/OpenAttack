from typing import List, Optional
import numpy as np
import copy

from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...text_process.tokenizer import Tokenizer, get_default_tokenizer
from ...attack_assist.substitute.word import WordSubstitute, get_default_substitute
from ...utils import get_language, check_language, language_by_name
from ...exceptions import WordNotInDictionaryException
from ...tags import Tag
from ...attack_assist.filter_words import get_default_filter_words


class PSOAttacker(ClassificationAttacker):
    @property
    def TAGS(self):
        return { self.__lang_tag, Tag("get_pred", "victim"), Tag("get_prob", "victim")}

    def __init__(self, 
            pop_size : int = 20,
            max_iters : int = 20,
            tokenizer : Optional[Tokenizer] = None,
            substitute : Optional[WordSubstitute] = None,
            filter_words : List[str] = None,
            lang = None
        ):
        """
        Word-level Textual Adversarial Attacking as Combinatorial Optimization. Yuan Zang, Fanchao Qi, Chenghao Yang, Zhiyuan Liu, Meng Zhang, Qun Liu and Maosong Sun. ACL 2020.
        `[pdf] <https://www.aclweb.org/anthology/2020.acl-main.540.pdf>`__
        `[code] <https://github.com/thunlp/SememePSO-Attack>`__

        Args:
            pop_size: Genetic algorithm popluation size. **Default:** 20
            max_iter: Maximum generations of pso algorithm. **Default:** 20
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
        
        if substitute is None:
            substitute = get_default_substitute(self.__lang_tag)
        self.substitute = substitute

        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer

        self.pop_size = pop_size
        self.max_iters = max_iters
        if filter_words is None:
            filter_words = get_default_filter_words(self.__lang_tag)
        self.filter_words = set(filter_words)

        check_language([self.tokenizer, self.substitute], self.__lang_tag)

    def attack(self, victim: Classifier, sentence, goal: ClassifierGoal):
        self.invoke_dict = {}
        x_orig = sentence.lower()
        
        x_orig = self.tokenizer.tokenize(x_orig)
        x_pos = list(map(lambda x: x[1], x_orig))
        x_orig = list(map(lambda x: x[0], x_orig))

        x_len = len(x_orig)
        neighbours_nums = [
            min(self.get_neighbour_num(word, pos),10) if word not in self.filter_words else 0
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
        pop = self.generate_population(x_orig, neighbours, w_select_probs, x_len)


        part_elites = pop
        if goal.targeted:
            all_elite_score = 100
            part_elites_scores = [100 for _ in range(self.pop_size)]
        else:
            all_elite_score = -1
            part_elites_scores = [-1 for _ in range(self.pop_size)]
        all_elite = pop[0]

        Omega_1 = 0.8
        Omega_2 = 0.2
        C1_origin = 0.8
        C2_origin = 0.2
        V = [np.random.uniform(-3, 3) for _ in range(self.pop_size)]
        V_P = [[V[t] for _ in range(x_len)] for t in range(self.pop_size)]
        for i in range(self.max_iters):
            pop_preds = self.predict_batch(victim, pop)
            pop_scores = pop_preds[:, goal.target]

            if goal.targeted:
                pop_ranks = np.argsort(pop_scores)[::-1]
                top_attack = pop_ranks[0]
                if np.max(pop_scores) > all_elite_score:
                    all_elite = pop[top_attack]
                    all_elite_score = np.max(pop_scores)
                for k in range(self.pop_size):
                    if pop_scores[k] > part_elites_scores[k]:
                        part_elites[k] = pop[k]
                        part_elites_scores[k] = pop_scores[k]
                if np.argmax(pop_preds[top_attack, :]) == goal.target:
                    return self.tokenizer.detokenize(pop[top_attack])
            else:
                pop_ranks = np.argsort(pop_scores)
                top_attack = pop_ranks[0]
                if np.min(pop_scores) < all_elite_score:
                    all_elite = pop[top_attack]
                    all_elite_score = np.min(pop_scores)
                for k in range(self.pop_size):
                    if pop_scores[k] < part_elites_scores[k]:
                        part_elites[k] = pop[k]
                        part_elites_scores[k] = pop_scores[k]
                if np.argmax(pop_preds[top_attack, :]) != goal.target:
                    return self.tokenizer.detokenize(pop[top_attack])
                
            Omega = (Omega_1 - Omega_2) * (self.max_iters - i) / self.max_iters + Omega_2
            C1 = C1_origin - i / self.max_iters * (C1_origin - C2_origin)
            C2 = C2_origin + i / self.max_iters * (C1_origin - C2_origin)
            for id in range(self.pop_size):

                for dim in range(x_len):
                    V_P[id][dim] = Omega * V_P[id][dim] + (1 - Omega) * (
                                self.equal(pop[id][dim], part_elites[id][dim]) + self.equal(pop[id][dim],
                                                                                            all_elite[dim]))
                turn_prob = [self.sigmod(V_P[id][d]) for d in range(x_len)]
                P1 = C1
                P2 = C2

                if np.random.uniform() < P1:
                    pop[id] = self.turn(part_elites[id], pop[id], turn_prob, x_len)
                if np.random.uniform() < P2:
                    pop[id] = self.turn(all_elite, pop[id], turn_prob, x_len)
            pop_preds = self.predict_batch(victim, pop)
            pop_scores = pop_preds[:, goal.target]
            if goal.targeted:
                pop_ranks = np.argsort(pop_scores)[::-1]
                top_attack = pop_ranks[0]
                if np.max(pop_scores) > all_elite_score:
                    all_elite = pop[top_attack]
                    all_elite_score = np.max(pop_scores)
                for k in range(self.pop_size):
                    if pop_scores[k] > part_elites_scores[k]:
                        part_elites[k] = pop[k]
                        part_elites_scores[k] = pop_scores[k]
                if np.argmax(pop_preds[top_attack, :]) == goal.target:
                    return self.tokenizer.detokenize( pop[top_attack] )
            else:
                pop_ranks = np.argsort(pop_scores)
                top_attack = pop_ranks[0]
                if np.min(pop_scores) < all_elite_score:
                    all_elite = pop[top_attack]
                    all_elite_score = np.min(pop_scores)
                for k in range(self.pop_size):
                    if pop_scores[k] < part_elites_scores[k]:
                        part_elites[k] = pop[k]
                        part_elites_scores[k] = pop_scores[k]
                if np.argmax(pop_preds[top_attack, :]) != goal.target:
                    return self.tokenizer.detokenize( pop[top_attack] )

            new_pop = []
            for x in pop:
                change_ratio = self.count_change_ratio(x, x_orig, x_len)
                p_change = 1 - 2 * change_ratio
                if np.random.uniform() < p_change:
                    tem = self.mutate( x, x_orig, neighbours, w_select_probs)
                    new_pop.append(tem)
                else:
                    new_pop.append(x)
            pop = new_pop

        return None #Failed

    def predict_batch(self, victim, sentences):

        return np.array([self.predict(victim, s) for s in sentences])

    def predict(self, victim, sentence):
        if tuple(sentence) in self.invoke_dict:
            return self.invoke_dict[tuple(sentence)]

        tem = victim.get_prob(self.make_batch([sentence]))[0]

        self.invoke_dict[tuple(sentence)] = tem
        return tem

    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new


    def generate_population(self, x_orig, neighbours_list, w_select_probs, x_len):
        pop = []
        x_len = w_select_probs.shape[0]
        for i in range(self.pop_size):
            r = np.random.choice(x_len, 1, p=w_select_probs)[0]
            replace_list = neighbours_list[r]
            sub = np.random.choice(replace_list, 1)[0]
            tem = self.do_replace(x_orig, r, sub)
            pop.append(tem)
        return pop


    def turn(self, x1, x2, prob, x_len):
        x_new = copy.deepcopy(x2)
        for i in range(x_len):
            if np.random.uniform() < prob[i]:
                x_new[i] = x1[i]
        return x_new

    def mutate(self, x, x_orig, neigbhours_list, w_select_probs):
        x_len = w_select_probs.shape[0]
        rand_idx = np.random.choice(x_len, 1,p=w_select_probs)[0]
        while x[rand_idx] != x_orig[rand_idx] and self.sum_diff(x_orig,x) < np.sum(np.sign(w_select_probs)):
            rand_idx = np.random.choice(x_len, 1,p=w_select_probs)[0]
        replace_list = neigbhours_list[rand_idx]
        sub_idx= np.random.choice(len(replace_list), 1)[0]
        new_x=copy.deepcopy(x)
        new_x[rand_idx]=replace_list[sub_idx]
        return new_x

    def sum_diff(self, x_orig, x_cur):
        ret = 0
        for wa, wb in zip(x_orig, x_cur):
            if wa != wb:
                ret += 1
        return ret

    def norm(self, n):

        tn = []
        for i in n:
            if i <= 0:
                tn.append(0)
            else:
                tn.append(i)
        s = np.sum(tn)
        if s == 0:
            for i in range(len(tn)):
                tn[i] = 1
            return [t / len(tn) for t in tn]
        new_n = [t / s for t in tn]

        return new_n


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


    def make_batch(self, sents):
        return [self.tokenizer.detokenize(sent) for sent in sents]

    def equal(self, a, b):
        if a == b:
            return -3
        else:
            return 3

    def sigmod(self, n):
        return 1 / (1 + np.exp(-n))

    def count_change_ratio(self, x, x_orig, x_len):
        change_ratio = float(np.sum(np.array(x) != np.array(x_orig))) / float(x_len)
        return change_ratio









