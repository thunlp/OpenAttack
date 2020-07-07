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
    "import_score_threshold": -1.,
    "sim_score_threshold": 0.5, 
    "sim_score_window": 15, 
    "synonym_num": 50,
    "processor": DefaultTextProcessor(),
    "substitute": None,
    "embedding_matrix": None,
    "vocab": None,
}

class TextFoolerAttacker(Attacker):
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
        x_copy = x_orig
        if target is None:
            targeted = False
            target = clsf.get_pred([x_orig])[0]  # calc x_orig's prediction
        else:
            targeted = True
        orig_probs = clsf.get_prob([x_orig])
        orig_label = clsf.get_pred([x_orig])
        orig_prob = orig_probs.max()
        x_orig = list(map(lambda x: x[0], self.config["processor"].get_tokens(x_orig)))

        len_text = len(x_orig)
        if len_text < self.config["sim_score_window"]:
            self.config["sim_score_threshold"] = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (self.config["sim_score_window"] - 1) // 2


        # get the pos and verb tense info
        pos_ls = list(map(lambda x: x[1], self.config["processor"].get_tokens(x_copy)))
        #pos_ls = criteria.get_pos(text_ls)

        # get importance score

        leave_1_texts = [x_orig[:ii] + ['<oov>'] + x_orig[min(ii + 1, len_text):] for ii in range(len_text)]
        leave_1_probs = clsf.get_prob([' '.join(sentence) for sentence in leave_1_texts])
        leave_1_probs_argmax = np.argmax(leave_1_probs, axis=-1)
        #import_scores = orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).astype(np.float64) * (
        #            leave_1_probs.max(axis=-1)[0] - orig_probs[:, leave_1_probs_argmax])

        import_scores = orig_prob - leave_1_probs[:, orig_label].squeeze() + (leave_1_probs_argmax != orig_label).astype(np.float64) * (
                    np.max(leave_1_probs, axis=-1) - orig_probs.squeeze()[leave_1_probs_argmax])



        # get words to perturb ranked by importance score for word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > self.config["import_score_threshold"] and x_orig[idx] not in self.config["skip_words"]:
                    words_perturb.append((idx, x_orig[idx]))
            except:
                print(idx, len(x_orig), import_scores.shape, x_orig, len(leave_1_texts))


        # find synonyms
        #words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words = [
            self.get_neighbours(word)
            if word not in self.config["skip_words"]
            else []
            for idx, word in words_perturb
        ]
        #synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, self.config["synonym_num"], 0.5)
        synonyms_all = []
        for idx, word in words_perturb:
            synonyms = synonym_words.pop(0)
            if synonyms:
                synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = x_orig[:]
        text_cache = text_prime[:]
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = clsf.get_prob([' '.join(sentence) for sentence in new_texts])

            # compute semantic similarity
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
            semantic_sims = self.semantic_sim([text_cache[text_range_min:text_range_max] for i in range(len(new_texts))],
                                       list(map(lambda x: x[text_range_min:text_range_max], new_texts)), 
                                       self.config["embedding_matrix"], self.config["vocab"])[0]

            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = orig_label != np.argmax(new_probs, axis=-1)
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= self.config["sim_score_threshold"])
            # prevent incompatible pos
            synonyms_pos_ls = [list(map(lambda x: x[1], self.config["processor"].get_tokens(' '.join(new_text[max(idx - 4, 0):idx + 5]))))[min(4, idx)]
                               if len(new_text) > 10 else list(map(lambda x: x[1], self.config["processor"].get_tokens(' '.join(new_text))))[idx] for new_text in new_texts]

            pos_mask = np.array(self.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                x_adv = detokenizer(text_prime)
                pred = clsf.get_pred([x_adv])
                if not targeted:
                    return (x_adv, pred[0])
                elif pred[0] == target:
                    return (x_adv, pred[0])
            else:
                #new_label_probs = new_probs[:, orig_label] + (semantic_sims < self.config["sim_score_threshold"]) + (1 - pos_mask).astype(np.float64)
                new_label_probs = new_probs[:, orig_label] + (semantic_sims < self.config["sim_score_threshold"])
                
                new_label_prob_min = np.min(new_label_probs, axis=0)[0]
                new_label_prob_argmin = np.argmin(new_label_probs, axis=0)[0]
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
            text_cache = text_prime[:]
        return None
            

    def get_neighbours(self, word):
        threshold = 0.5
        try:
            return list(
                map(
                    lambda x: x[0],
                    self.config["substitute"](word, threshold=threshold)[1 : self.config["synonym_num"] + 1],
                )
            )
        except WordNotInDictionaryException:
            return []

    def semantic_sim(self, sents1, sents2, embedding_matrix, vocab):
        embeds1 = self.embedding_process(sents1, embedding_matrix, vocab)
        embeds2 = self.embedding_process(sents2, embedding_matrix, vocab)
        norm1 = np.linalg.norm(embeds1, ord=2, axis=2)
        norm2 = np.linalg.norm(embeds2, ord=2, axis=2)
        cosine_similarities = np.sum(norm1 * norm2, axis=1)
        clip_cosine_similarities = np.clip(cosine_similarities, -1.0, 1.0)
        scores = 1.0 - np.arccos(clip_cosine_similarities)
        return scores


    def embedding_process(self, input_, embedding_matrix, vocab):
        seqs = []
        max_len = len(input_[0])
        for sentence in input_:
            seq = []
            for word in sentence:
                if len(seq) < max_len:
                    if word in vocab:
                        seq.append(vocab[word])
                    else:
                        seq.append(0)
            while len(seq) < max_len:
                seq.append(0)
            seqs.append(seq)
        embedding_dim = embedding_matrix.shape[1]
        embeds = np.zeros(shape=((len(seqs), max_len, embedding_dim)))
        for i in range(len(seqs)):
            for j in range(max_len):
                embeds[i, j, :] = embedding_matrix[seqs[i][j]]
        return embeds

    def pos_filter(self, ori_pos, new_pos_list):
        same = [True if ori_pos == new_pos or (set([ori_pos, new_pos]) <= set(['NOUN', 'VERB']))
                else False for new_pos in new_pos_list]
        return same