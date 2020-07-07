from ..attacker import Attacker
from ..text_processors import DefaultTextProcessor
from ..data_manager import DataManager
import random
# from spacy.lang.en import English
# import nltk
# from nltk.tokenize import TreebankWordTokenizer
# from spacy.lang.en import English


DEFAULT_CONFIG = {

}


class TextBuggerAttacker(Attacker):
    def __init__(self, **kwargs):
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        self.nlp = DataManager.load("NLTKSentTokenizer")
        self.textprocesser = DefaultTextProcessor()
        self.glove_vectors = None
        # self.nlp = English()
        # self.glove_vectors = DataManager.load("GloveVector")
        # self.treebank = TreebankWordTokenizer()
        # self.nlp = English()
        # self.treebank = DataManager.load("TREEBANK")

    def __call__(self, clsf, x_orig, target=None):
        """
        * **clsf** : **Classifier** .
        * **x_orig** : Input sentence.
        """
        if target is None:
            target = clsf.get_pred([x_orig])[0]
        # x = self.treebank.tokenize(x_orig)  # tokenize
        x = self.tokenize(x_orig)
        sentences_of_doc = self.get_sentences(x)
        ranked_sentences = self.rank_sentences(sentences_of_doc, clsf, target)
        x_prime = x.copy()

        for sentence_index in ranked_sentences:
            ranked_words = self.get_word_importances(sentences_of_doc[sentence_index], clsf, target)
            for word in ranked_words:
                bug = self.selectBug(word, x_prime, clsf)
                x_prime = self.replaceWithBug(x_prime, word, bug)
                # x_prime_sentence = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(x_prime)
                x_prime_sentence = " ".join(x_prime)
                prediction = clsf.get_pred([x_prime_sentence])[0]

                # if self.getSemanticSimilarity(x, x_prime, self.epsilon) <= self.epsilon:
                #    return None  # elelelelelel
                if prediction != target:
                    return " ".join(x_prime), prediction
        # print("None found")
        return x_orig, target

    def get_sentences(self, x):
        # original_review = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(x)
        # self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        # doc = self.nlp(original_review)
        original_review = " ".join(x)
        doc = self.nlp(original_review)
        sentences = [sent.strip() for sent in doc]
        return sentences

    def rank_sentences(self, sentences, clsf, target_all):
        import torch
        
        map_sentence_to_loss = {}  # 与原文不同
        for i in range(len(sentences)):
            target = clsf.get_pred([sentences[i]])[0]
            if target != target_all:
                continue
            with torch.no_grad():
                tempoutput = torch.from_numpy(clsf.get_prob(sentences[i]))
            # map_sentence_to_loss[i] = F.nll_loss(tempoutput, target_all, reduce=False)
            softmax = torch.nn.Softmax(dim=1)
            nll_lossed = -1 * torch.log(softmax(tempoutput))[0][target_all].item()
            map_sentence_to_loss[i] = nll_lossed
        sentences_sorted_by_loss = {k: v for k, v in sorted(map_sentence_to_loss.items(), key=lambda item: -item[1], reverse=True)}
        return sentences_sorted_by_loss

    def get_word_importances(self, sentence, clsf, target):
        import torch

        # sentence_tokens = self.treebank.tokenize(sentence)
        sentence_tokens = self.tokenize(sentence)
        word_losses = {}
        for curr_token in sentence_tokens:
            sentence_tokens_without = [token for token in sentence_tokens if token != curr_token]
            # sentence_without = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(sentence_tokens_without)
            sentence_without = " ".join(sentence_tokens_without)
            with torch.no_grad():
                tempoutput = torch.from_numpy(clsf.get_prob(sentence_without))
            # word_losses[curr_token] = F.nll_loss(tempoutput, target, reduce=False)
            softmax = torch.nn.Softmax(dim=1)
            nll_lossed = -1 * torch.log(softmax(tempoutput))[0][target].item()
            word_losses[curr_token] = nll_lossed
        word_losses = {k: v for k, v in sorted(word_losses.items(), key=lambda item: -item[1], reverse=True)}
        return word_losses

    def getSemanticSimilarity(x, x_prime, epsilon):
        # to be continue
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
        import torch

        # x_prime_sentence = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(x_prime)
        # candidate_sentence = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(candidate)
        x_prime_sentence = " ".join(x_prime)
        candidate_sentence = " ".join(candidate)
        target = clsf.get_pred([x_prime_sentence])[0]
        with torch.no_grad():
            tempoutput = torch.from_numpy(clsf.get_prob(candidate_sentence))
        # x_prime_loss = F.nll_loss(tempoutput, target, reduce=False)
        softmax = torch.nn.Softmax(dim=1)
        nll_lossed = -1 * torch.log(softmax(tempoutput))[0][target].item()
        x_prime_loss = nll_lossed
        return x_prime_loss

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
        return bugs

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
        # tokens = sent.strip().split()
        tokens = list(map(lambda x: x[0], self.textprocesser.get_tokens(sent)))
        return tokens
