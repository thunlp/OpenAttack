from ..attacker import Attacker
from ..text_processors import DefaultTextProcessor
from ..data_manager import DataManager
from ..substitutes import CounterFittedSubstitute
from ..exceptions import WordNotInDictionaryException
import random
import numpy as np


DEFAULT_CONFIG = {
    "blackbox": True,
    "textprocessor": DefaultTextProcessor()
}


class TextBuggerAttacker(Attacker):
    def __init__(self, **kwargs):
        """
        :param bool blackbox: Classifier Capacity. True-probability; False-grad. **Default:** True.

        :Data Requirements: :py:data:`.TProcess.NLTKSentTokenizer`
        :Classifier Capacity: Blind or Probability

        TEXTBUGGER: Generating Adversarial Text Against Real-world Applications. Jinfeng Li, Shouling Ji, Tianyu Du, Bo Li, Ting Wang. NDSS 2019.
        `[pdf] <https://arxiv.org/pdf/1812.05271.pdf>`__
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        self.nlp = DataManager.load("TProcess.NLTKSentTokenizer")
        self.textprocessor = self.config["textprocessor"]
        self.counterfit = CounterFittedSubstitute()
        self.glove_vectors = None

    def __call__(self, clsf, x_orig, target=None):
        """
        * **clsf** : **Classifier** .
        * **x_orig** : Input sentence.
        """
        y_orig = clsf.get_pred([x_orig])[0]
        x = self.tokenize(x_orig)
        sentences_of_doc = self.get_sentences(x)
        ranked_sentences = self.rank_sentences(sentences_of_doc, clsf, y_orig)
        x_prime = x.copy()

        for sentence_index in ranked_sentences:
            if self.config["blackbox"] is True:
                ranked_words = self.get_word_importances(sentences_of_doc[sentence_index], clsf, y_orig)
            else:
                ranked_words = self.get_w_word_importances(sentences_of_doc[sentence_index], clsf, y_orig)
            for word in ranked_words:
                bug = self.selectBug(word, x_prime, clsf)
                x_prime = self.replaceWithBug(x_prime, word, bug)
                x_prime_sentence = self.textprocessor.detokenizer(x_prime)
                prediction = clsf.get_pred([x_prime_sentence])[0]

                if target is None:
                    if prediction != y_orig:
                        return self.textprocessor.detokenizer(x_prime), prediction
                else:
                    if int(prediction) is int(target):
                        return self.textprocessor.detokenizer(x_prime), prediction
        return None

    def get_sentences(self, x):
        return [self.textprocessor.detokenizer(x)]

    def rank_sentences(self, sentences, clsf, target_all):
        map_sentence_to_loss = {}  # 与原文不同
        for i in range(len(sentences)):
            y_orig = clsf.get_pred([sentences[i]])[0]
            if y_orig != target_all:
                continue
            tempoutput = clsf.get_prob([sentences[i]])[0]
            ret = max(tempoutput[target_all], 1e-3)
            map_sentence_to_loss[i] = -np.log(ret)
        sentences_sorted_by_loss = {k: v for k, v in sorted(map_sentence_to_loss.items(), key=lambda item: -item[1], reverse=True)}
        return sentences_sorted_by_loss

    def get_word_importances(self, sentence, clsf, y_orig):
        sentence_tokens = self.tokenize(sentence)
        word_losses = {}
        for curr_token in sentence_tokens:
            sentence_tokens_without = [token for token in sentence_tokens if token != curr_token]
            sentence_without = self.textprocessor.detokenizer(sentence_tokens_without)
            tempoutput = clsf.get_prob([sentence_without])[0]
            ret = max(tempoutput[y_orig], 1e-3)
            word_losses[curr_token] = -np.log(ret)
        word_losses = {k: v for k, v in sorted(word_losses.items(), key=lambda item: -item[1], reverse=True)}
        return word_losses

    def get_w_word_importances(self, sentence, clsf, y_orig):  # white
        from collections import OrderedDict

        sentence_tokens = self.tokenize(sentence)
        prob, grad = clsf.get_grad([sentence_tokens], [y_orig])
        grad = grad[0]
        dist = []
        for i in range(len(grad)):
            dist.append(0.0)
            for j in range(len(grad[i])):
                dist[i] += grad[i][j] * grad[i][j]
            dist[i] = np.sqrt(dist[i])
        word_losses = OrderedDict()
        for i, curr_token in enumerate(sentence_tokens):
            if i < len(dist):
                word_losses[curr_token] = dist[i]
            else:
                word_losses[curr_token] = 0
        word_losses = {k: v for k, v in sorted(word_losses.items(), key=lambda item: -item[1], reverse=True)}
        return word_losses

    def getSemanticSimilarity(x, x_prime, epsilon):
        return epsilon + 1

    def selectBug(self, original_word, x_prime, clsf):
        bugs = self.generateBugs(original_word, self.glove_vectors)
        max_score = float('-inf')
        best_bug = original_word
        bug_tracker = {}
        for bug_type, b_k in bugs.items():
            candidate_k = self.getCandidate(original_word, b_k, x_prime)
            score_k = self.getScore(candidate_k, x_prime, clsf)
            if score_k > max_score:
                best_bug = b_k
                max_score = score_k
            bug_tracker[b_k] = score_k
        return best_bug

    def getCandidate(self, original_word, new_bug, x_prime):
        tokens = x_prime
        new_tokens = [new_bug if x == original_word else x for x in tokens]
        return new_tokens

    def getScore(self, candidate, x_prime, clsf):
        x_prime_sentence = self.textprocessor.detokenizer(x_prime)
        candidate_sentence = self.textprocessor.detokenizer(candidate)
        y_orig = clsf.get_pred([x_prime_sentence])[0]
        tempoutput = clsf.get_prob([candidate_sentence])[0]
        ret = max(tempoutput[y_orig], 1e-3)
        return -np.log(ret)

    def replaceWithBug(self, x_prime, x_i, bug):
        tokens = x_prime
        new_tokens = [bug if x == x_i else x for x in tokens]
        return new_tokens

    def generateBugs(self, word, glove_vectors, sub_w_enabled=False, typo_enabled=False):
        bugs = {"insert": word, "delete": word, "swap": word, "sub_C": word, "sub_W": word}
        if len(word) <= 2:
            return bugs
        bugs["insert"] = self.bug_insert(word)
        bugs["delete"] = self.bug_delete(word)
        bugs["swap"] = self.bug_swap(word)
        bugs["sub_C"] = self.bug_sub_C(word)
        bugs["sub_W"] = self.bug_sub_W(word)
        return bugs

    def bug_sub_W(self, word):
        try:
            res = self.counterfit.__call__(word, pos=None, threshold=0.5)
            if len(res) == 0:
                return word
            return res[0][0]
        except WordNotInDictionaryException:
            return word

    def bug_insert(self, word):
        if len(word) >= 6:
            return word
        res = word
        point = random.randint(1, len(word) - 1)
        res = res[0:point] + " " + res[point:]
        return res

    def bug_delete(self, word):
        res = word
        point = random.randint(1, len(word) - 2)
        res = res[0:point] + res[point + 1:]
        return res

    def bug_swap(self, word):
        if len(word) <= 4:
            return word
        res = word
        points = random.sample(range(1, len(word) - 1), 2)
        a = points[0]
        b = points[1]

        res = list(res)
        w = res[a]
        res[a] = res[b]
        res[b] = w
        res = ''.join(res)
        return res

    def bug_sub_C(self, word):
        res = word
        key_neighbors = self.get_key_neighbors()
        point = random.randint(0, len(word) - 1)

        if word[point] not in key_neighbors:
            return word
        choices = key_neighbors[word[point]]
        subbed_choice = choices[random.randint(0, len(choices) - 1)]
        res = list(res)
        res[point] = subbed_choice
        res = ''.join(res)

        return res

    def get_key_neighbors(self):
        # By keyboard proximity
        neighbors = {
            "q": "was", "w": "qeasd", "e": "wrsdf", "r": "etdfg", "t": "ryfgh", "y": "tughj", "u": "yihjk",
            "i": "uojkl", "o": "ipkl", "p": "ol",
            "a": "qwszx", "s": "qweadzx", "d": "wersfxc", "f": "ertdgcv", "g": "rtyfhvb", "h": "tyugjbn",
            "j": "yuihknm", "k": "uiojlm", "l": "opk",
            "z": "asx", "x": "sdzc", "c": "dfxv", "v": "fgcb", "b": "ghvn", "n": "hjbm", "m": "jkn"
        }
        # By visual proximity
        neighbors['i'] += '1'
        neighbors['l'] += '1'
        neighbors['z'] += '2'
        neighbors['e'] += '3'
        neighbors['a'] += '4'
        neighbors['s'] += '5'
        neighbors['g'] += '6'
        neighbors['b'] += '8'
        neighbors['g'] += '9'
        neighbors['q'] += '9'
        neighbors['o'] += '0'

        return neighbors

    def tokenize(self, sent):
        tokens = list(map(lambda x: x[0], self.textprocessor.get_tokens(sent)))
        return tokens
