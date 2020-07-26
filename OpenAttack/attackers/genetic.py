import numpy as np
from ..text_processors import DefaultTextProcessor
from ..substitutes import CounterFittedSubstitute
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
    "neighbour_threshold": 0.5,
    "top_n1": 20,
    "processor": DefaultTextProcessor(),
    "substitute": None,
}


class GeneticAttacker(Attacker):
    def __init__(self, **kwargs):
        """
        :param list skip_words: A list of words which won't be replaced during the attack. **Default:** A list of words that is most frequently used.
        :param int pop_size: Genetic algorithm popluation size. **Default:** 20
        :param int max_iter: Maximum generations of genetic algorithm. **Default:** 20
        :param float neighbour_threshold: Threshold used in substitute module. **Default:** 0.5
        :param int top_n1: Maximum candidates of word substitution. **Default:** 20
        :param TextProcessor processor: Text processor used in this attacker. **Default:** :any:`DefaultTextProcessor`
        :param WordSubstitute substitute: Substitute method used in this attacker. **Default:** :any:`CounterFittedSubstitute`

        :Classifier Capacity: Probability

        Generating Natural Language Adversarial Examples. Moustafa Alzantot, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani Srivastava, Kai-Wei Chang. EMNLP 2018.
        `[pdf] <https://www.aclweb.org/anthology/D18-1316.pdf>`__
        `[code] <https://github.com/nesl/nlp_adversarial_examples>`__
        
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        if self.config["substitute"] is None:
            self.config["substitute"] = CounterFittedSubstitute()

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
        x_pos =  list(map(lambda x: x[1], x_orig))
        x_orig = list(map(lambda x: x[0], x_orig))

        x_len = len(x_orig)
        neighbours_nums = [
            self.get_neighbour_num(word, pos) if word not in self.config["skip_words"] else 0
            for word, pos in zip(x_orig, x_pos)
        ]
        neighbours = [
            self.get_neighbours(word, pos, self.config["top_n1"])
            if word not in self.config["skip_words"]
            else []
            for word, pos in zip(x_orig, x_pos)
        ]

        if np.sum(neighbours_nums) == 0:
            return None
        w_select_probs = neighbours_nums / np.sum(neighbours_nums)

        pop = [  # generate population
            self.perturb(
                clsf, x_orig, x_orig, neighbours, w_select_probs, target, targeted
            )
            for _ in range(self.config["pop_size"])
        ]
        for i in range(self.config["max_iters"]):
            pop_preds = clsf.get_prob(self.make_batch(pop))
            if targeted:
                top_attack = np.argmax(pop_preds[:, target])
                if np.argmax(pop_preds[top_attack, :]) == target:
                    return self.config["processor"].detokenizer(pop[top_attack]), target
            else:
                top_attack = np.argmax(-pop_preds[:, target])
                if np.argmax(pop_preds[top_attack, :]) != target:
                    return (
                        self.config["processor"].detokenizer(pop[top_attack]),
                        np.argmax(pop_preds[top_attack, :]),
                    )

            pop_scores = pop_preds[:, target]
            if targeted:
                pass
            else:
                pop_scores = 1.0 - pop_scores

            if np.sum(pop_scores) == 0:
                return None
            pop_scores = pop_scores / np.sum(pop_scores)

            elite = [pop[top_attack]]
            parent_indx_1 = np.random.choice(
                self.config["pop_size"], size=self.config["pop_size"] - 1, p=pop_scores
            )
            parent_indx_2 = np.random.choice(
                self.config["pop_size"], size=self.config["pop_size"] - 1, p=pop_scores
            )
            childs = [
                self.crossover(pop[p1], pop[p2])
                for p1, p2 in zip(parent_indx_1, parent_indx_2)
            ]
            childs = [
                self.perturb(
                    clsf, x_cur, x_orig, neighbours, w_select_probs, target, targeted
                )
                for x_cur in childs
            ]
            pop = elite + childs

        return None  # Failed

    def get_neighbour_num(self, word, pos):
        threshold = self.config["neighbour_threshold"]
        cnt = 0
        try:
            return len(self.config["substitute"](word, pos, threshold=threshold))
        except WordNotInDictionaryException:
            return 0

    def get_neighbours(self, word, pos, num):
        threshold = self.config["neighbour_threshold"]
        try:
            return list(
                map(
                    lambda x: x[0],
                    self.config["substitute"](word, pos, threshold=threshold)[:num],
                )
            )
        except WordNotInDictionaryException:
            return []

    def select_best_replacements(
        self, clsf, indx, neighbours, x_cur, x_orig, target, targeted
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

        pred_scores = clsf.get_prob(self.make_batch(new_list))[:, target]
        if targeted:
            new_scores = pred_scores[:-1] - pred_scores[-1]
        else:
            new_scores = pred_scores[-1] - pred_scores[:-1]

        if np.max(new_scores) > 0:
            return new_list[np.argmax(new_scores)]
        else:
            return x_cur

    def make_batch(self, sents):
        return [self.config["processor"].detokenizer(sent) for sent in sents]

    def perturb(
        self, clsf, x_cur, x_orig, neighbours, w_select_probs, target, targeted
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
            clsf, mod_idx, neighbours[mod_idx], x_cur, x_orig, target, targeted
        )

    def crossover(self, x1, x2):
        ret = []
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                ret.append(x1[i])
            else:
                ret.append(x2[i])
        return ret
