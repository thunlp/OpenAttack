import numpy as np
import copy
from ..text_processors import DefaultTextProcessor
from ..substitutes import HowNetSubstitute
from ..exceptions import WordNotInDictionaryException
from ..utils import check_parameters
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
    "pop_size": 20,
    "max_iters": 20,
    "processor": DefaultTextProcessor(),
    "substitute": None,
}


class PSOAttacker(Attacker):
    def __init__(self, **kwargs):
        """
        :param list skip_words: A list of words which won't be replaced during the attack. **Default:** A list of words that is most frequently used.
        :param int pop_size: Genetic algorithm popluation size. **Default:** 20
        :param int max_iter: Maximum generations of pso algorithm. **Default:** 20
        :param TextProcessor processor: Text processor used in this attacker. **Default:** :any:`DefaultTextProcessor`
        :param WordSubstitute substitute: Substitute method used in this attacker. **Default:** :any:`hownet`

        :Data Requirements: :py:data:`.AttackAssist.HowNet` :py:data:`.TProcess.NLTKWordNet`
        :Package Requirements: * **OpenHowNet**

        :Classifier Capacity: Probability
        
        Word-level Textual Adversarial Attacking as Combinatorial Optimization. Yuan Zang, Fanchao Qi, Chenghao Yang, Zhiyuan Liu, Meng Zhang, Qun Liu and Maosong Sun. ACL 2020.

        `[pdf] <https://www.aclweb.org/anthology/2020.acl-main.540.pdf>`__
        `[code] <https://github.com/thunlp/SememePSO-Attack>`__

        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        if self.config["substitute"] is None:
            self.config["substitute"] = HowNetSubstitute()

        check_parameters(DEFAULT_CONFIG.keys(), self.config)

    def __call__(self, clsf, x_orig, target=None):
        """
        * **clsf** : **Classifier** .
        * **x_orig** : Input sentence.
        """
        x_orig = x_orig.lower()
        if target is None:
            targeted = False
            target = clsf.get_pred([x_orig])[0]  # calc x_orig's prediction
        else:
            targeted = True
        x_orig = self.config["processor"].get_tokens(x_orig)
        x_pos = list(map(lambda x: x[1], x_orig))
        x_orig = list(map(lambda x: x[0], x_orig))

        x_len = len(x_orig)
        neighbours_nums = [
            self.get_neighbour_num(word, pos) if word not in self.config["skip_words"] else 0
            for word, pos in zip(x_orig, x_pos)
        ]
        neighbours = [
            self.get_neighbours(word, pos)
            if word not in self.config["skip_words"]
            else []
            for word, pos in zip(x_orig, x_pos)
        ]

        if np.sum(neighbours_nums) == 0:
            return None

        pop = self.generate_population(
                clsf, x_orig, neighbours,neighbours_nums,x_len, target, targeted)
        part_elites = copy.deepcopy(pop)
        pop_preds = clsf.get_prob(self.make_batch(pop))
        pop_scores = pop_preds[:, target]
        part_elites_scores = pop_scores
        if targeted:
            all_elite_score = np.max(pop_scores)
            pop_ranks = np.argsort(pop_scores)[::-1]
            top_attack = pop_ranks[0]
            all_elite = pop[top_attack]
            if np.argmax(pop_preds[top_attack, :]) == target:
                return self.config["processor"].detokenizer(pop[top_attack]), target
        else:
            all_elite_score = np.min(pop_scores)
            pop_ranks = np.argsort(pop_scores)
            top_attack = pop_ranks[0]
            all_elite = pop[top_attack]
            if np.argmax(pop_preds[top_attack, :]) != target:
                return (
                    self.config["processor"].detokenizer(pop[top_attack]),
                    np.argmax(pop_preds[top_attack, :]),
                )
        Omega_1 = 0.8
        Omega_2 = 0.2
        C1_origin = 0.8
        C2_origin = 0.2
        V = [np.random.uniform(-3, 3) for rrr in range(self.config["pop_size"])]
        V_P = [[V[t] for rrr in range(x_len)] for t in range(self.config["pop_size"])]
        for i in range(self.config["max_iters"]):
            Omega = (Omega_1 - Omega_2) * (self.config["max_iters"] - i) / self.config["max_iters"] + Omega_2
            C1 = C1_origin - i / self.config["max_iters"] * (C1_origin - C2_origin)
            C2 = C2_origin + i / self.config["max_iters"] * (C1_origin - C2_origin)
            for id in range(self.config["pop_size"]):

                for dim in range(x_len):
                    V_P[id][dim] = Omega * V_P[id][dim] + (1 - Omega) * (
                                self.equal(pop[id][dim], part_elites[id][dim]) + self.equal(pop[id][dim],
                                                                                            all_elite[dim]))
                turn_prob = [self.sigmod(V_P[id][d]) for d in range(x_len)]
                P1 = C1
                P2 = C2
                # P1=self.sigmod(P1)
                # P2=self.sigmod(P2)

                if np.random.uniform() < P1:
                    pop[id] = self.turn(part_elites[id], pop[id], turn_prob, x_len)
                if np.random.uniform() < P2:
                    pop[id] = self.turn(all_elite, pop[id], turn_prob, x_len)
            pop_preds = clsf.get_prob(self.make_batch(pop))
            pop_scores = pop_preds[:, target]
            if targeted:
                pop_ranks = np.argsort(pop_scores)[::-1]
                top_attack = pop_ranks[0]
                print('\t\t', i, ' -- ', 'before mutation', pop_scores[top_attack])
                if np.argmax(pop_preds[top_attack, :]) == target:
                    return self.config["processor"].detokenizer(pop[top_attack]), target
            else:
                pop_ranks = np.argsort(pop_scores)
                top_attack = pop_ranks[0]
                print('\t\t', i, ' -- ', 'before mutation', pop_scores[top_attack])
                if np.argmax(pop_preds[top_attack, :]) != target:
                    return (
                        self.config["processor"].detokenizer(pop[top_attack]),
                        np.argmax(pop_preds[top_attack, :]),
                    )
            new_pop = []
            for x in pop:
                change_ratio = self.count_change_ratio(x, x_orig, x_len)
                p_change = 1 - 2 * change_ratio
                if np.random.uniform() < p_change:
                    new_h, new_w_list = self.gen_h_score(clsf,x_len, target,targeted,neighbours_nums, neighbours, x)
                    new_pop.append(self.mutate(x, new_h, new_w_list))
                else:
                    new_pop.append(x)
            pop = new_pop
            pop_preds = clsf.get_prob(self.make_batch(pop))
            pop_scores = pop_preds[:, target]
            if targeted:
                pop_ranks = np.argsort(pop_scores)[::-1]
                top_attack = pop_ranks[0]
                print('\t\t', i, ' -- ', 'after mutation', pop_scores[top_attack])
                if np.argmax(pop_preds[top_attack, :]) == target:
                    return self.config["processor"].detokenizer(pop[top_attack]), target
            else:
                pop_ranks = np.argsort(pop_scores)
                top_attack = pop_ranks[0]
                print('\t\t', i, ' -- ', 'after mutation', pop_scores[top_attack])
                if np.argmax(pop_preds[top_attack, :]) != target:
                    return (
                        self.config["processor"].detokenizer(pop[top_attack]),
                        np.argmax(pop_preds[top_attack, :]),
                    )
            if targeted:
                for k in range(self.config["pop_size"]):
                    if pop_scores[k] > part_elites_scores[k]:
                        part_elites[k] = pop[k]
                        part_elites_scores[k] = pop_scores[k]
                elite = pop[top_attack]
                if np.max(pop_scores) > all_elite_score:
                    all_elite = elite
                    all_elite_score = np.max(pop_scores)
            else:
                for k in range(self.config["pop_size"]):
                    if pop_scores[k] < part_elites_scores[k]:
                        part_elites[k] = pop[k]
                        part_elites_scores[k] = pop_scores[k]
                elite = pop[top_attack]
                if np.min(pop_scores) < all_elite_score:
                    all_elite = elite
                    all_elite_score = np.min(pop_scores)
        return None #Failed
    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def mutate(self, x_cur, w_select_probs, w_list):
        x_len = w_select_probs.shape[0]
        # print('w_select_probs:',w_select_probs)
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        return self.do_replace(x_cur, rand_idx, w_list[rand_idx])

    def generate_population(self, clsf,x_orig,neighbours_list,neighbours_len,x_len, target, targeted):
        h_score, w_list = self.gen_h_score(clsf,x_len, target,targeted,neighbours_len, neighbours_list, x_orig)
        return [self.mutate(x_orig, h_score, w_list) for _ in
                range(self.config["pop_size"])]

    def turn(self, x1, x2, prob, x_len):
        x_new = copy.deepcopy(x2)
        for i in range(x_len):
            if np.random.uniform() < prob[i]:
                x_new[i] = x1[i]
        return x_new

    def gen_most_change(self,clsf, pos, x_cur, target,targeted, replace_list):
        new_list = []
        rep_words = []
        for word in replace_list:
            if word != x_cur[pos]:
                new_list.append(self.do_replace(x_cur,pos,word))
                rep_words.append(word)
        if len(new_list)==0:
            return 0,x_cur[pos]
        new_list.append(x_cur)

        pred_scores = clsf.get_prob(self.make_batch(new_list))[:, target]
        if targeted:
            new_scores = pred_scores[:-1] - pred_scores[-1]
        else:
            new_scores = pred_scores[-1] - pred_scores[:-1]

        return np.max(new_scores), new_list[np.argsort(new_scores)[-1]][pos]

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

    def gen_h_score(self, clsf,x_len, target,targeted, neighbours_len, neigbhours_list, x_now):

        w_list = []
        prob_list = []
        for i in range(x_len):
            if neighbours_len[i] == 0:
                w_list.append(x_now[i])
                prob_list.append(0)
                continue
            p, w = self.gen_most_change(clsf,i, x_now, target,targeted,neigbhours_list[i])
            w_list.append(w)
            prob_list.append(p)

        prob_list = self.norm(prob_list)
        # print('neighbours_len:',neighbours_len)
        # print('prob_list:',prob_list)

        h_score = prob_list
        h_score = np.array(h_score)
        return h_score, w_list
    def get_neighbour_num(self, word, pos):
        try:
            return len(self.config["substitute"](word, pos))
        except WordNotInDictionaryException:
            return 0

    def get_neighbours(self, word, pos):
        try:
            return list(
                map(
                    lambda x: x[0],
                    self.config["substitute"](word, pos),
                )
            )
        except WordNotInDictionaryException:
            return []


    def make_batch(self, sents):
        return [self.config["processor"].detokenizer(sent) for sent in sents]

    def equal(self, a, b):
        if a == b:
            return -3
        else:
            return 3

    def sigmod(self, n):
        return 1 / (1 + np.exp(-n))

    def count_change_ratio(self, x, x_orig, x_len):
        change_ratio = float(np.sum(x != x_orig)) / float(x_len)
        return change_ratio










