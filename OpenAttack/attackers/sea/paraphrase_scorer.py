import time
import os
import copy
import numpy as np
from . import onmt_model
from . import onmt
import collections
import operator
import editdistance
import sys
import itertools
from itertools import zip_longest as zip_longest

DEFAULT_TO_PATHS = ['data/TranslationModels/english_french_model_acc_71.05_ppl_3.71_e13.pt', 'data/TranslationModels/english_portuguese_model_acc_70.75_ppl_4.32_e13.pt']
DEFAULT_BACK_PATHS = ['data/TranslationModels/french_english_model_acc_68.51_ppl_4.43_e13.pt', 'data/TranslationModels/portuguese_english_model_acc_69.93_ppl_5.04_e13.pt']

def choose_forward_translation(sentence, to_translator, back_translator, n=5):

    translations = to_translator.translate([sentence], n_best=n,
                                           return_from_mapping=True)[0]
    mappings = [x[2] for x in translations if x[0]]
    translations = [x[0] for x in translations if x[0]]
    scores = [back_translator.score(x, [sentence])[0] for x in translations]
    return translations[np.argmax(scores)], mappings[np.argmax(scores)]

def normalize_ll(x):

    max_ = x.max()
    b = np.exp(x - max_)
    return b / b.sum()

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    if n > flat.shape[0]:
        indices = np.array(range(flat.shape[0]), dtype='int')
        return np.unravel_index(indices, ary.shape)
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


class ParaphraseScorer(object):
    def __init__(self,
                 to_paths=DEFAULT_TO_PATHS,
                 back_paths=DEFAULT_BACK_PATHS,
                 gpu_id=0,
                 cuda=True):
        self.to_translators = []

        self.back_translators = []
        for f in to_paths:
            translator = onmt_model.OnmtModel(f, gpu_id, cuda)
            self.to_translators.append(translator)

        for f in back_paths:
            translator = onmt_model.OnmtModel(f, gpu_id, cuda)
            self.back_translators.append(translator)
        self.build_common_vocabs()
        self.last = None

    def build_common_vocabs(self):
        self.global_itos = []
        self.global_stoi = {}
        self.vocab_mappers = []
        self.back_vocab_mappers = []
        self.vocab_unks = []
        back_vocab_mappers = []
        for t in self.back_translators:
            vocab = t.translator.fields['tgt'].vocab
            mapper = []
            back_mapper = {}
            for i, w in enumerate(vocab.itos):
                if w not in self.global_stoi:
                    self.global_stoi[w] = len(self.global_stoi)
                    self.global_itos.append(w)
                mapper.append(self.global_stoi[w])
                back_mapper[self.global_stoi[w]] = i
            self.vocab_mappers.append(np.array(mapper))
            back_vocab_mappers.append(back_mapper)
        for t, m, back_mapper in zip(self.back_translators, self.vocab_mappers,
                                     back_vocab_mappers):
            unks = np.array(
                    list(set(range(len(self.global_itos))).difference(m)))
            for u in unks:
                back_mapper[u] = onmt.IO.UNK
            bm = np.zeros(len(self.global_itos), dtype=int)
            for b, v in back_mapper.items():
                bm[b] = v
            self.back_vocab_mappers.append(bm)
            self.vocab_unks.append(unks)


    def nearby_distribution(self, sentence, weight_by_edit_distance=False, **kwargs):
        paraphrases = self.generate_paraphrases(sentence, **kwargs)
        if not paraphrases:
            return paraphrases
        others = [x[0] for x in paraphrases]
        n_scores = normalize_ll(np.array([x[1] for x in paraphrases]))
        if weight_by_edit_distance:
            return self.weight_by_edit_distance(sentence, list(zip(others, n_scores)))
        return sorted(zip(others, n_scores), key=lambda x:x[1], reverse=True)

    def weight_by_edit_distance(self, sentence, distribution):

        others = [x[0] for x in distribution]
        n_scores = np.array([x[1] for x in distribution])
        import editdistance
        orig = onmt_model.clean_text(sentence)
        orig_score = self.score_sentences(orig, [orig])[0]
        orig = orig.split()

        n_scores = np.minimum(0, n_scores - orig_score)

        distances = np.array([editdistance.eval(orig, x.split()) for x in others])

        logkernel = lambda d, k: .5 * -(d**2) / (k**2)


        n_scores = n_scores + logkernel(distances, 3)


        n_scores = normalize_ll(n_scores)

        return sorted(zip(others, n_scores), key=lambda x: x[1], reverse=True)

   
    def generate_paraphrases(self, sentence, topk=10, threshold=None, edit_distance_cutoff=None, penalize_unks=True, frequent_ngrams=None):

        assert threshold or topk
        encoder_states = []
        contexts = []
        decoder_states = []
        new_sizes = []
        src_examples = []
        PROFILING = False
        mappings = []
        to_add = -10000 if penalize_unks else 0

        for to, back in zip(self.to_translators, self.back_translators):
            translation, mapping = choose_forward_translation(sentence, to, back, n=5)
            mappings.append(mapping)
            encStates, context, decStates, src_example = back.get_init_states(translation)
            src_examples.append(src_example)
            encoder_states.append(encStates)
            contexts.append(context)
            decoder_states.append(decStates)
            new_sizes.append([1])
        orig_score = self.score_sentences(sentence, [sentence])[0]
        if threshold:
            threshold = threshold + orig_score

        orig = onmt_model.clean_text(sentence)
        output = dict([(orig, orig_score)])
        orig_ids = np.array([self.global_stoi[onmt.IO.BOS_WORD]] + [self.global_stoi[x] if x in self.global_stoi else onmt.IO.UNK for x in orig.split()])
        orig_words = [onmt.IO.BOS_WORD] + orig.split()
        orig_itoi = {}
        orig_stoi = {}
        for i, w in zip(orig_ids, orig_words):
            if w not in orig_stoi:
                idx = len(orig_stoi)
                orig_stoi[w] = idx
                if i not in orig_itoi:
                    orig_itoi[i] = idx

        not_in_sentence = np.array(
            list(set(self.global_stoi.values()).difference(
                set(list(orig_itoi.keys()) + [onmt.IO.UNK]))))
        mapped_orig = [orig_stoi[x] for x in orig_words]
        if frequent_ngrams is not None:
            import difflib
            new_f = set()
            new_f.add(tuple())
            for f, v in frequent_ngrams.items():
                for t in v:
                    new_f.add(tuple(sorted([orig_stoi[x] for x in t])))
        prev = [[self.global_stoi[onmt.IO.BOS_WORD]]]
        prev_scores = [0]
        prev_distance_rep = [[orig_itoi[prev[0][0]]]]
        idxs = [self.global_stoi[onmt.IO.BOS_WORD]]
        new_sizes = [1]
        prev_unks = [['']]
        
        while prev:
            orig_time = time.time()
            global_scores = np.zeros((len(prev), (len(self.global_itos))))

            all_stuff = zip(
                self.back_translators, self.vocab_mappers,
                self.back_vocab_mappers, self.vocab_unks, contexts,
                decoder_states, encoder_states, src_examples, mappings)
            new_decoder_states = []
            new_attns = []
            unk_scores = []

            for (b, mapper, back_mapper, unks, context,
                 decStates, encStates, srcz, mapping) in all_stuff:
                idx = [int(back_mapper[i]) for i in idxs]
                out, decStates, attn = b.advance_states(
                    encStates, context, decStates, idx, new_sizes)

                new_decoder_states.append(decStates)
                new_attns.append(attn)
                attenz = attn['std'].data[0].cpu().numpy()
                chosen = np.argmax(attenz, axis=1)
                for r, ch in enumerate(chosen):
                    ch = mapping[ch]
                    if ch in b.vocab().stoi:
                        ind = b.vocab().stoi[ch]
                        out[r, ind] = max(out[r, ind], out[r, onmt.IO.UNK])
                    elif ch in self.global_stoi:
                        ind = self.global_stoi[ch]
                        global_scores[r, ind] -= to_add
                        
                unk_scores.append(out[:, onmt.IO.UNK])
                global_scores[:, mapper] += out

                if unks.shape[0]:
                    global_scores[:, unks] += to_add + out[:, onmt.IO.UNK][:, np.newaxis]

            decoder_states = new_decoder_states
            global_scores /= float(len(self.back_translators))

            unk_scores = [normalize_ll(x) for x in np.array(unk_scores).T]
            
            new_prev = []
            new_prev_distance_rep = []
            new_prev_unks = []
            new_prev_scores = []
            new_sizes = []
            new_origins = []
            idxs = []

            new_scores = global_scores + np.array(prev_scores)[:, np.newaxis]

            def get_possibles(opcodes):
                possibles = [tuple()]
                for tag, i1, i2, j1, j2 in opcodes:
                    if tag == 'equal':
                        continue
                    if tag == 'insert':
                        cha = range(j1, j2)
                        if len(cha) == 2:
                            possibles.append([i1 - 1])
                            possibles.append([i1])
                        if len(cha) > 2:
                            possibles = []
                            break
                        if len(cha) == 1:
                            possibles.append([i1 - 1, i1])
                    if tag == 'replace':
                        for i1, j1 in zip_longest(range(i1, i2), range(j1, j2)):
                            if i1 is None:
                                i1 = i2
                                possibles.append([i1 - 1, i1])
                            elif j1 is None:
                                possibles.append([i1])
                            else:
                                possibles.append([i1])
                    if tag == 'delete':
                        for i1 in range(i1, i2):
                            possibles.append([i1])
                if len(possibles) > 1:
                    possibles.pop(0)
                return possibles
            if frequent_ngrams is not None:
                for i, p_rep in enumerate(prev_distance_rep):
                    for idx, v in orig_itoi.items():

                        if idx == onmt.IO.UNK:
                            continue
                        candidate = p_rep + [v]
                        a = difflib.SequenceMatcher(a = mapped_orig[:len(candidate)], b=candidate)
                        possibles = get_possibles(a.get_opcodes())
                        if len(possibles) == 1 and possibles[0] == tuple():
                            continue
                        if not np.any([x in new_f for x in itertools.product(*possibles)]):
                            new_scores[i, idx] = -100000
                candidate = p_rep + [-1]
                a = difflib.SequenceMatcher(a = mapped_orig[:len(candidate)], b=candidate)
                possibles = get_possibles(a.get_opcodes())
                if not np.any([x in new_f for x in itertools.product(*possibles)]):
                    new_scores[i, not_in_sentence] = -10000
            if edit_distance_cutoff is not None:
                for i, p_rep in enumerate(prev_distance_rep):
                    for idx, v in orig_itoi.items():

                        if idx == onmt.IO.UNK:
                            continue
                        candidate = p_rep + [v]
                        distance = editdistance.eval(candidate, mapped_orig[:len(candidate)])

                        if distance > edit_distance_cutoff:
                            new_scores[i, idx] = -100000
                candidate = p_rep + [-1]
                distance = editdistance.eval(candidate, mapped_orig[:len(candidate)])
                if distance > edit_distance_cutoff:
                    new_scores[i, not_in_sentence] = -10000
            if threshold:
                where = np.where(new_scores > threshold)
                if topk:
                    largest = largest_indices(new_scores[where], topk)[0]
                    where = (where[0][largest], where[1][largest])
            else:
                where = largest_indices(new_scores, topk)

            tmp = np.argsort(where[0])
            where = (where[0][tmp], where[1][tmp])

            if (edit_distance_cutoff is not None and
                    len(prev[0]) < len(orig_ids) and
                    orig_ids[len(prev[0])] not in where[1][where[0] == 0]):
                where = (np.hstack(([0], where[0])),
                         np.hstack(([orig_ids[len(prev[0])]], where[1])))

            for i, j in zip(*where):
                if j == self.global_stoi[onmt.IO.EOS_WORD]:
                    words = [self.global_itos[x] if x != onmt.IO.UNK
                             else prev_unks[i][k]
                             for k, x in enumerate(prev[i][1:], start=1)]
                    new = " ".join(words)
                    if new not in output:
                        output[new] = new_scores[i, j]
                    else:
                        output[new] = max(output[new], new_scores[i, j])
                    continue
                new_origins.append(i)
                new_unk = '<unk>'
                if j == onmt.IO.UNK:
                    new_unk_scores = collections.defaultdict(lambda: 0)
                    for x, src, mapping, score_weight in zip(new_attns, src_examples, mappings, unk_scores[i]):
                        attn = x['std'].data[0][i]
                        for zidx, (word, score) in enumerate(zip(src, attn)):
                            word = mapping[zidx]
                            new_unk_scores[word] += score * score_weight
                    new_unk = max(new_unk_scores.items(),
                                  key=operator.itemgetter(1))[0]


                if edit_distance_cutoff is not None:
                    distance_rep = orig_itoi[j] if j in orig_itoi else -1
                    if j == onmt.IO.UNK:
                        distance_rep = (orig_stoi[new_unk] if new_unk in orig_stoi
                                        else -1)
                    new_prev_distance_rep.append(prev_distance_rep[i] +
                                                 [distance_rep])
                new_prev.append(prev[i] + [j])
                new_prev_unks.append(prev_unks[i] + [new_unk])
                new_prev_scores.append(new_scores[i, j])
                idxs.append(j)


            new_sizes = np.bincount(new_origins, minlength=len(prev))
            new_sizes = [int(x) for x in new_sizes]
            prev = new_prev
            prev_unks = new_prev_unks
            prev_distance_rep = new_prev_distance_rep
            prev_scores = new_prev_scores
            if topk and len(output) > topk:
                break
            for z, s, unks in zip(prev, prev_scores, prev_unks):
                words = [self.global_itos[x] if x != onmt.IO.UNK else 'UNK'+ unks[k] for k, x in enumerate(z)]


        return sorted(output.items(), key=lambda x:x[1], reverse=True)




    def score_sentences(self, original_sentence, other_sentences, relative_to_original=False, verbose=False):
        memoized_stuff = self.last == original_sentence
        if relative_to_original:
            other_sentences = [original_sentence] + other_sentences
        all_scores = []
        if not memoized_stuff:
            self.last = original_sentence
            self.memoized = {}
            self.memoized['translation'] = []
        if verbose:
            score_to_print = []
        for k, (to, back, mapper, back_mapper, unks) in enumerate(
                zip(self.to_translators, self.back_translators,
                    self.vocab_mappers, self.back_vocab_mappers,
                    self.vocab_unks)):
            if memoized_stuff:
                translation, mapping = self.memoized['translation'][k]
            else:
                translation, mapping = choose_forward_translation(original_sentence, to, back,
                                                         n=5)
                self.memoized['translation'].append((translation, mapping))
            this_scores = []
            if verbose:
                scorezz = []
            for s in other_sentences:
                s = onmt_model.clean_text(s)
                orig_ids = np.array([self.global_stoi[onmt.IO.BOS_WORD]] + [self.global_stoi[x] if x in self.global_stoi else onmt.IO.UNK for x in s.split()] + [self.global_stoi[onmt.IO.EOS_WORD]])
                score = 0.
                encStates, context, decStates, src_example = (
                    back.get_init_states(translation))
                for i, n in zip(orig_ids, orig_ids[1:]):
                    idx = int(back_mapper[i])
                    n = int(back_mapper[n])
                    out, decStates, attn = back.advance_states(encStates, context,
                                                               decStates, [idx], [1])
                    attenz = attn['std'].data[0].cpu().numpy()
                    ch = np.argmax(attenz, axis=1)[0]
                    ch = mapping[ch]
                    if ch in back.vocab().stoi:
                        ind = back.vocab().stoi[ch]
                        out[0, ind] = max(out[0, ind], out[0, onmt.IO.UNK])
                    score += out[0][n]
                    if verbose:
                        scorezz.append( (self.global_itos[n], out[0][n]))
                this_scores.append(score)
                if verbose:
                    scorezz.append(('\n', score))
            if verbose:
                score_to_print.append(scorezz)
            all_scores.append(this_scores)
        scores = np.mean(all_scores, axis=0)
        if verbose:
            for z in zip(*score_to_print):
                print('%-10s'% z[0][0], end=' ')
                for a in z:
                    print('%.2f' % a[1], end=' ')
                print()
        if relative_to_original:
            scores = (scores - scores[0])[1:]
        return scores

    def weighted_scores(self, original_sentence, other_sentences,
                        relative_to_original=False):
        scores = self.score_sentences(original_sentence, other_sentences)
        self_scores = []
        for s in other_sentences:
            self_scores.append(self.score_sentences(s, [s])[0])
        self_scores = np.array(self_scores)
        elementwise_max = np.maximum(scores, self_scores)
        n_scores = np.exp(scores - elementwise_max)
        n_self_scores = np.exp(self_scores - elementwise_max)
        return np.log(n_scores / (n_scores + n_self_scores))
