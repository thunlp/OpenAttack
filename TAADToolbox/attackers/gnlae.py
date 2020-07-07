import numpy as np
from ..text_processors import DefaultTextProcessor
from ..substitutes import CounterFittedSubstitute
from ..exceptions import WordNotInDictionaryException
from ..utils import check_parameters, detokenizer
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


class GNLAEAttacker(Attacker):
    def __init__(self, **kwargs):
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
        x_orig = list(map(lambda x: x[0], self.config["processor"].get_tokens(x_orig)))

        x_len = len(x_orig)
        neighbours_nums = [
            self.get_neighbour_num(word) if word not in self.config["skip_words"] else 0
            for word in x_orig
        ]
        neighbours = [
            self.get_neighbours(word, self.config["top_n1"])
            if word not in self.config["skip_words"]
            else []
            for word in x_orig
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
                    return detokenizer(pop[top_attack]), target
            else:
                top_attack = np.argmax(-pop_preds[:, target])
                if np.argmax(pop_preds[top_attack, :]) != target:
                    return (
                        detokenizer(pop[top_attack]),
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

    def get_neighbour_num(self, word):
        threshold = self.config["neighbour_threshold"]
        cnt = 0
        try:
            return len(self.config["substitute"](word, threshold=threshold))
        except WordNotInDictionaryException:
            return 0

    def get_neighbours(self, word, num):
        threshold = self.config["neighbour_threshold"]
        try:
            return list(
                map(
                    lambda x: x[0],
                    self.config["substitute"](word, threshold=threshold)[:num],
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
        return [detokenizer(sent) for sent in sents]

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
