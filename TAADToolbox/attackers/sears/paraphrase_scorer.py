import time
import os
import copy
import numpy as np
from . import onmt_model
#import onmt_model
import onmt
import collections
import operator
import editdistance
import sys
import itertools
from itertools import zip_longest as zip_longest


# DEFAULT_TO_PATHS = ['/home/marcotcr/OpenNMT-py/trained_models/english_french_model_acc_70.61_ppl_3.73_e13.pt', '/home/marcotcr/OpenNMT-py/trained_models/english_german_model_acc_58.34_ppl_7.82_e13.pt', '/home/marcotcr/OpenNMT-py/trained_models/english_portuguese_model_acc_70.90_ppl_4.28_e13.pt']
# DEFAULT_BACK_PATHS = ['/home/marcotcr/OpenNMT-py/trained_models/french_english_model_acc_68.83_ppl_4.43_e13.pt', '/home/marcotcr/OpenNMT-py/trained_models/german_english_model_acc_57.23_ppl_10.00_e13.pt', '/home/marcotcr/OpenNMT-py/trained_models/portuguese_english_model_acc_69.78_ppl_5.05_e13.pt']
DEFAULT_TO_PATHS = ['data/TranslationModels/english_french_model_acc_71.05_ppl_3.71_e13.pt', 'data/TranslationModels/english_portuguese_model_acc_70.75_ppl_4.32_e13.pt']
DEFAULT_BACK_PATHS = ['data/TranslationModels/french_english_model_acc_68.51_ppl_4.43_e13.pt', 'data/TranslationModels/portuguese_english_model_acc_69.93_ppl_5.04_e13.pt']

def choose_forward_translation(sentence, to_translator, back_translator, n=5):
    # chooses the to_translation that gives the best back_score to
    # sentence given back_translation
    translations = to_translator.translate([sentence], n_best=n,
                                           return_from_mapping=True)[0]
    mappings = [x[2] for x in translations if x[0]]
    translations = [x[0] for x in translations if x[0]]
    # translations = [x[0] for x in
    #                 to_translator.translate([sentence], n_best=n)[0] if x[0]]
    scores = [back_translator.score(x, [sentence])[0] for x in translations]
    return translations[np.argmax(scores)], mappings[np.argmax(scores)]

def normalize_ll(x):
    # normalizes vector of log likelihoods
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
                 gpu_id=1):
        print('GPU ID', gpu_id)
        self.to_translators = []
        # self.to_scorers = []
        self.back_translators = []
        for f in to_paths:
            translator = onmt_model.OnmtModel(f, gpu_id)
            self.to_translators.append(translator)
            # self.to_scorers.append(translator)
        for f in back_paths:
            translator = onmt_model.OnmtModel(f, gpu_id)
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
        # Distribution is a list of (text, weight) tuples, unnormalized
        others = [x[0] for x in distribution]
        n_scores = np.array([x[1] for x in distribution])
        import editdistance
        orig = onmt_model.clean_text(sentence)
        orig_score = self.score_sentences(orig, [orig])[0]
        orig = orig.split()
        # print(orig_score)
        n_scores = np.minimum(0, n_scores - orig_score)
        # n_scores = n_scores - orig_score
        distances = np.array([editdistance.eval(orig, x.split()) for x in others])

        logkernel = lambda d, k: .5 * -(d**2) / (k**2)

        # This is equivalent to multiplying the prediction probability by the exponential kernel on the distance
        n_scores = n_scores + logkernel(distances, 3)
        # zeros = np.where(distances == 0)[0]
        # print(n_scores)
        # print(distances[np.argsort(distances)])
        # print(distances)
        # print(n_scores)
        # w = np.log2(distances + 1)
        # w = (distances + 1) ** (1./2)
        # print(n_scores.argmax())
        # print(w)
        # n_scores = w * n_scores
        # This is equivalent to dividing the predict proba by the distance + 1
        # n_scores =  n_scores - np.log(distances + 1)
        # print(distances[n_scores.argmax()])
        # n_scores[zeros] = -99999999
        # print(distances[n_scores.argmax()])

        n_scores = normalize_ll(n_scores)
        # print(n_scores.argmax())

        # TODO: achar funcao pra weight by edit distance
        # n_scores = n_scores / (distances + 1)
        # n_scores = n_scores / n_scores.sum()
        return sorted(zip(others, n_scores), key=lambda x: x[1], reverse=True)

    def suggest_next(self, words, idx, in_between=False, run_through=False, topk=10, threshold=None,
                     original_sentence=None, only_local_score=False):
        # TODO: This is outdated

        to_add = -10000
        memoized_stuff = self.last == original_sentence and original_sentence is not None
        # print('suggest_next', words[:idx], memoized_stuff)
        if not memoized_stuff:
            self.last = original_sentence
            self.memoized = {}
            self.memoized['translation'] = []
        sentence = (' '.join(words) if original_sentence is None
                    else original_sentence)
        words_after = words[idx:] if in_between else words[idx + 1:]
        words = words[:idx]
        print(words)
        print(words_after)
        orig_ids = np.array([self.global_stoi[onmt.IO.BOS_WORD]] + [self.global_stoi[x] if x in self.global_stoi else onmt.IO.UNK for x in words])
        global_scores = np.zeros(len(self.global_itos))
        last_scores = np.zeros(len(self.global_itos))
        unk_scores = []
        if threshold:
            threshold *= len(self.back_translators)
        attns = []
        src_examples = []
        dec_states = []
        enc_states = []
        contexts = []
        mappings = []
        for k, (to, back, mapper, back_mapper, unks) in enumerate(
                zip(self.to_translators, self.back_translators,
                    self.vocab_mappers, self.back_vocab_mappers,
                    self.vocab_unks)):
            if memoized_stuff:
                translation, mapping = self.memoized['translation'][k]
                mappings.append(mapping)
            else:
                translation, mapping = choose_forward_translation(sentence, to, back,
                                                         n=5)
                mappings.append(mapping)
                self.memoized['translation'].append((translation, mapping))
            encStates, context, decStates, src_example = (
                back.get_init_states(translation))
            src_examples.append(src_example)
            # print()
            # print(k)
            a = 0
            for i, n in zip(orig_ids, orig_ids[1:]):
                idx = int(back_mapper[i])
                n = int(back_mapper[n])
                out, decStates, attn = back.advance_states(encStates, context,
                                                           decStates, [idx], [1])
                attenz = attn['std'].data[0].cpu().numpy()
                chosen = np.argmax(attenz, axis=1)
                for r, ch in enumerate(chosen):
                    ch = mapping[ch]
                    if ch in back.vocab().stoi:
                        # print("YOO")
                        ind = back.vocab().stoi[ch]
                        # print("prev", out[r, ind])
                        out[r, ind] = max(out[r, ind], out[r, onmt.IO.UNK])
                        # print("aft", out[r, ind])
                    elif ch in self.global_stoi:
                        # print(ch)
                        ind = self.global_stoi[ch]
                        global_scores[ind] -= to_add
                global_scores[mapper] += out[0][n]
                a += out[0][n]
                # print(n, out[0][n])
                if unks.shape[0]:
                    global_scores[unks] += to_add + out[0, n]
            # print(np.argsort(out[0])[-5:])
            # print( 'g1', global_scores[63441])
            print('a', a)
            idx = int(back_mapper[orig_ids[-1]])
            out, decStates, attn = back.advance_states(encStates, context,
                                                       decStates, [idx], [1])
            attenz = attn['std'].data[0].cpu().numpy()
            chosen = np.argmax(attenz, axis=1)
            for r, ch in enumerate(chosen):
                ch = mapping[ch]
                if ch in back.vocab().stoi:
                    # print("YOO")
                    ind = back.vocab().stoi[ch]
                    # print("prev", out[r, ind])
                    out[r, ind] = max(out[r, ind], out[r, onmt.IO.UNK])
                    # print("aft", out[r, ind])
                elif ch in self.global_stoi:
                    # print(ch)
                    ind = self.global_stoi[ch]
                    global_scores[ind] -= to_add
                    last_scores[ind] -= to_add
            if unks.shape[0]:
                global_scores[unks] += to_add + out[0, onmt.IO.UNK]
                last_scores[unks] += to_add + out[0, onmt.IO.UNK]
            global_scores[mapper] += out[0]
            last_scores[mapper] += out[0]
            unk_scores.append(out[0, onmt.IO.UNK])
            attns.append(attn)
            dec_states.append(decStates)
            contexts.append(context)
            enc_states.append(encStates)
            # print( 'g2', global_scores[63441])
        unk_scores = normalize_ll(np.array(unk_scores))
        new_unk_scores = collections.defaultdict(lambda: 0)
        for x, src, mapping, score_weight in zip(attns, src_examples, mappings, unk_scores):
            # print(src)
            attn = x['std'].data[0][0]
            # TODO: Should we only allow unks here? We are
            # currently weighting based on the original score, but
            # this makes it so one always chooses the unk.
            for zidx, (word, score) in enumerate(zip(src, attn)):
                word = mapping[zidx]
                new_unk_scores[word] += score * score_weight
        # print(sorted(new_unk_scores.items(), key=lambda x:x[1], reverse=True))
        new_unk = max(new_unk_scores.items(),
                      key=operator.itemgetter(1))[0]
        # if new_unk in self.global_stoi:
        #     global_scores[onmt.IO.UNK] = global_scores[self.global_stoi[new_unk]]
        #     last_scores[onmt.IO.UNK] = last_scores[self.global_stoi[new_unk]]
        picked = largest_indices(global_scores, topk)[0]
        # if new_unk in self.global_stoi and onmt.IO.UNK in picked:
        #     picked[picked == onmt.IO.UNK] = self.global_stoi[new_unk]
        if threshold:
            to_keep = np.where(global_scores[picked] >= threshold)[0]
            to_delete = np.where(global_scores[picked] < threshold)[0]
            picked = picked[to_keep]
            print(picked.shape)
            topk = picked.shape[0]
            if not topk:
                return []
        # global_scores = np.repeat(global_scores[np.newaxis, :], len(picked),
        #                           axis=0)
        # print(last_scores[picked])
        # print( 'g', global_scores[picked])
        if run_through:
            orig_ids = np.array([self.global_stoi[x] if x in self.global_stoi else onmt.IO.UNK for x in words_after] +
                                [self.global_stoi[onmt.IO.EOS_WORD]])
            for to, back, mapper, back_mapper, encStates, context, decStates, mapping in zip(
                    self.to_translators, self.back_translators, self.vocab_mappers, self.back_vocab_mappers,
                    enc_states, contexts, dec_states, mappings):
                if not picked.shape[0]:
                    break
                idx = [int(back_mapper[x]) for x in picked]
                # print(idx)
                # print(idx)
                # print([self.global_itos[x] for x in picked])
                # idx = int(back_mapper[x])
                n = int(back_mapper[orig_ids[0]])
                # print(n, back.vocab().itos[n])
                out, decStates, attn = back.advance_states(encStates, context,
                                                           decStates, idx, [len(idx)])

                attenz = attn['std'].data[0].cpu().numpy()
                chosen = np.argmax(attenz, axis=1)
                for r, ch in enumerate(chosen):
                    ch = mapping[ch]
                    if ch in back.vocab().stoi:
                        # print("YOO")
                        ind = back.vocab().stoi[ch]
                        # print("prev", out[r, ind])
                        out[r, ind] = max(out[r, ind], out[r, onmt.IO.UNK])
                # print(n, out[:, n])
                global_scores[picked] += out[:, n]
                sizes = [1 for _ in range(topk)]
                if threshold:
                    to_keep = np.where(global_scores[picked] >= threshold)[0]
                    to_delete = np.where(global_scores[picked] < threshold)[0]
                    picked = picked[to_keep]
                    print(picked.shape)
                    for x in to_delete:
                        sizes[x] = 0
                    topk = picked.shape[0]
                    if not topk:
                        break
                # global_scores[back_mapper] += out[:, n]
                for i, next_ in zip(orig_ids, orig_ids[1:]):
                    idx = [int(back_mapper[i]) for _ in range(topk)]
                    n = int(back_mapper[next_])
                    out, decStates, attn = back.advance_states(encStates, context,
                                                               decStates, idx, sizes)
                    attenz = attn['std'].data[0].cpu().numpy()
                    chosen = np.argmax(attenz, axis=1)
                    for r, ch in enumerate(chosen):
                        ch = mapping[ch]
                        if ch in back.vocab().stoi:
                            # print("YOO")
                            ind = back.vocab().stoi[ch]
                            # print("prev", out[r, ind])
                            out[r, ind] = max(out[r, ind], out[r, onmt.IO.UNK])
                            # print("aft", out[r, ind])
                # print(np.argsort(out[0])[-5:])
                    global_scores[picked] += out[:, n]
                    # print(n, out[:, n])
                    sizes = [1 for _ in range(topk)]
                    if threshold:
                        to_keep = np.where(global_scores[picked] >= threshold)[0]
                        to_delete = np.where(global_scores[picked] < threshold)[0]
                        # print(picked.shape)
                        picked = picked[to_keep]
                        # print(picked.shape)
                        topk = picked.shape[0]
                        for x in to_delete:
                            sizes[x] = 0
                        # print(topk)
                        if not topk:
                            break
        global_scores /= len(self.back_translators)
        last_scores /= len(self.back_translators)
        # TODO: there may be duplicates because of new_Unk
        if only_local_score:
            ret = [(self.global_itos[z], last_scores[z]) if z != onmt.IO.UNK else (new_unk, last_scores[z]) for z in picked if self.global_itos[z] != onmt.IO.EOS_WORD]
        else:
            ret = [(self.global_itos[z], global_scores[z]) if z != onmt.IO.UNK else (new_unk, global_scores[z]) for z in picked if self.global_itos[z] != onmt.IO.EOS_WORD]
        return sorted(ret, key=lambda x: x[1], reverse=True)
        return global_scores, new_unk
        print()
        print(list(reversed([self.global_itos[x] for x in np.argsort(global_scores)[-100:]])))
        pass
    '''def suggest_in_between(self, words, idxs_middle, topk=10, threshold=None,
                     original_sentence=None, max_inserts=4, ignore_set=set(),
                     return_full_texts=False, orig_score=0, verbose=False):
        # TODO: This is outdated

        run_through = True
        to_add = -10000
        memoized_stuff = self.last == original_sentence and original_sentence is not None
        # print('suggest_next', words[:idx], memoized_stuff)
        if not memoized_stuff:
            self.last = original_sentence
            self.memoized = {}
            self.memoized['translation'] = []
        sentence = (' '.join(words) if original_sentence is None
                    else original_sentence)
        words_after = words[idxs_middle[-1] + 1:]
        words_between = words[idxs_middle[0]:idxs_middle[1] + 1]
        words = words[:idxs_middle[0]]
        words_before = words
        # print(words)
        # print(words_between)
        # print(words_after)
        max_iters = max_inserts + idxs_middle[1] - idxs_middle[0] + 1
        out_scores = {}
        orig_ids = np.array([self.global_stoi[onmt.IO.BOS_WORD]] + [self.global_stoi[x] if x in self.global_stoi else onmt.IO.UNK for x in words])
        after_ids = np.array([self.global_stoi[x] if x in self.global_stoi else onmt.IO.UNK for x in words_after] +
                                [self.global_stoi[onmt.IO.EOS_WORD]])
        mid_ids = np.array([self.global_stoi[x] if x in self.global_stoi else onmt.IO.UNK for x in words_between])
        unk_scores = []
        if threshold:
            orig_threshold = threshold
        attns = []
        src_examples = []
        decoder_states = []
        encoder_states = []
        contexts = []
        mappings = []
        prev_scores = 0
        feed_original = 0
        in_between = 0
        mid_score = 0
        for k, (to, back, mapper, back_mapper, unks) in enumerate(
                zip(self.to_translators, self.back_translators,
                    self.vocab_mappers, self.back_vocab_mappers,
                    self.vocab_unks)):
            if memoized_stuff:
                translation, mapping = self.memoized['translation'][k]
                mappings.append(mapping)
            else:
                translation, mapping = choose_forward_translation(sentence, to, back,
                                                         n=5)
                mappings.append(mapping)
                self.memoized['translation'].append((translation, mapping))
            encStates, context, decStates, src_example = (
                back.get_init_states(translation))
            src_examples.append(src_example)
            # print()
            # Feed in the original input
            tz = time.time()
            for i, n in zip(orig_ids, orig_ids[1:]):
                idx = int(back_mapper[i])
                n = int(back_mapper[n])
                out, decStates, attn = back.advance_states(encStates, context,
                                                           decStates, [idx], [1])
                attenz = attn['std'].data[0].cpu().numpy()
                chosen = np.argmax(attenz, axis=1)
                for r, ch in enumerate(chosen):
                    ch = mapping[ch]
                    if ch in back.vocab().stoi:
                        # print("YOO")
                        ind = back.vocab().stoi[ch]
                        # print("prev", out[r, ind])
                        out[r, ind] = max(out[r, ind], out[r, onmt.IO.UNK])
                        # print("aft", out[r, ind])
                prev_scores += out[0][n]
            mid_score += prev_scores
            feed_original += time.time() - tz
            decoder_states.append(decStates)
            contexts.append(context)
            encoder_states.append(encStates)
            # print("MID IDS", mid_ids)
            onmt_model.transform_dec_states(decStates, [1])
            decStates = copy.deepcopy(decStates)
            for i, n in zip([orig_ids[-1]] + list(mid_ids), list(mid_ids) + [after_ids[0]]):
                # print('mid', i, n)
                idx = int(back_mapper[i])
                n = int(back_mapper[n])
                out, decStates, attn = back.advance_states(encStates, context,
                                                           decStates, [idx], [1])
                attenz = attn['std'].data[0].cpu().numpy()
                chosen = np.argmax(attenz, axis=1)
                for r, ch in enumerate(chosen):
                    ch = mapping[ch]
                    if ch in back.vocab().stoi:
                        # print("YOO")
                        ind = back.vocab().stoi[ch]
                        # print("prev", out[r, ind])
                        out[r, ind] = max(out[r, ind], out[r, onmt.IO.UNK])
                        # print("aft", out[r, ind])
                mid_score += out[0][n]
                # print("INcreasing mid")
        prev = [[]]
        prev_scores = [prev_scores / float(len(self.back_translators))]
        mid_score = mid_score / float(len(self.back_translators))
        if verbose:
            print('MID', mid_score)
        if threshold:
            threshold = mid_score + threshold
        # print(prev_scores)
        prev_unks = [[]]
        new_sizes = [1]
        idxs = [orig_ids[-1]]
        current_iter = 0
        # print(list(reversed([(self.global_itos[x], global_scores[0][x]) for x in np.argsort(global_scores[0])[-10:]])))
        going_after = 0
        while prev and current_iter < max_iters + 1:
            if verbose:
                print('iter', current_iter, topk)
            current_iter += 1
            global_scores = np.zeros((len(prev), (len(self.global_itos))))
            all_stuff = zip(
                self.back_translators, self.vocab_mappers,
                self.back_vocab_mappers, self.vocab_unks, contexts,
                decoder_states, encoder_states, src_examples, mappings)
            new_decoder_states = []
            new_attns = []
            unk_scores = []
            tz = time.time()
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
            new_prev_unks = []
            new_prev_scores = []
            new_sizes = []
            new_origins = []
            idxs = []
            new_scores = global_scores + np.array(prev_scores)[:, np.newaxis]
            # best = new_scores.max()
            # if threshold:
                # threshold = mid_score + orig_threshold
                # threshold = best + orig_threshold
                # threshold = orig_score + orig_threshold
                # print('best', best)
                # print('new thresh', threshold)
                # print(threshold == best + orig_threshold)
            if threshold:
                # print(threshold)
                where = np.where(new_scores > threshold)
                if topk:
                    largest = largest_indices(new_scores[where], topk)[0]
                    where = (where[0][largest], where[1][largest])
            else:
                where = largest_indices(new_scores, topk)
            # print('best', new_scores[where[0][0], where[1][0]], new_scores.max())
            tmp = np.argsort(where[0])
            where = (where[0][tmp], where[1][tmp])
            # print(where)
            new_this_round = []
            new_origins_this_round = []
            to_add = time.time() - tz
            in_between += time.time() - tz
            if verbose:
                print('in', to_add,  in_between, threshold)
                print(where[0].shape)
            for i, j in zip(*where):
                if j == after_ids[0]:
                    words = [self.global_itos[x] if x != onmt.IO.UNK
                             else prev_unks[i][k]
                             for k, x in enumerate(prev[i], start=0)]
                    new_full = ' '.join(words_before + words + words_after)
                    new = ' '.join(words)
                    if return_full_texts:
                        new = new_full
                    if new_full in ignore_set:
                        continue
                    # return
                    if new not in out_scores or new_scores[i, j] > out_scores[new]:
                        out_scores[new] = new_scores[i, j]
                        new_this_round.append(new)
                        new_origins_this_round.append(i)
                    # if topk:
                    #     topk -= 1
                    continue
                if j == self.global_stoi[onmt.IO.EOS_WORD]:
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
                # print (' '.join(self.global_itos[x] for x in prev[i][1:]))
                new_prev.append(prev[i] + [j])
                new_prev_unks.append(prev_unks[i] + [new_unk])
                new_prev_scores.append(new_scores[i, j])
                # print(i, j, new_scores[i,j])
                idxs.append(j)
            # print('newog', new_origins_this_round)
            # print(new_sizes)
            # print('idxs')
            # print(idxs)
            # for i, p in enumerate(prev):
            #     print(i, end= ' ')
            #     print([self.global_itos[x] for x in p], end=' ')
            #     print(list(reversed([(self.global_itos[x], new_scores[i][x]) for x in np.argsort(new_scores[i])[-10:]])))
            new_sizes = np.bincount(new_origins, minlength=len(prev))
            new_sizes = [int(x) for x in new_sizes]
            nsizes_this_round = np.bincount(new_origins_this_round, minlength=len(prev))
            nsizes_this_round = [int(x) for x in nsizes_this_round]
            # global_scores = np.zeros((len(prev), (len(self.global_itos))))
            zaaa = time.time()
            ndec_states = copy.deepcopy(decoder_states)
            all_stuff = zip(
                self.back_translators, self.vocab_mappers,
                self.back_vocab_mappers, self.vocab_unks, contexts,
                ndec_states, encoder_states, mappings)
            if len(new_this_round):
                # print(out_scores)
                for (b, mapper, back_mapper, unks, context,
                     decStates, encStates, mapping) in all_stuff:
                    nsizes = nsizes_this_round
                    # print('new b')
                    for i, next_ in zip(after_ids, after_ids[1:]):
                        # print(self.global_itos[i], self.global_itos[next_])
                        idx = [int(back_mapper[i]) for _ in new_this_round]
                        # print(len(nsizes_this_round))
                        # print(len(idx), sum(nsizes_this_round))
                        # print(nsizes)
                        n = int(back_mapper[next_])
                        # decStates = copy.deepcopy(decStates)
                        out, decStates, attn = b.advance_states(
                            encStates, context, decStates, idx, nsizes)
                        attenz = attn['std'].data[0].cpu().numpy()
                        chosen = np.argmax(attenz, axis=1)
                        for r, ch in enumerate(chosen):
                            ch = mapping[ch]
                            if ch in b.vocab().stoi:
                                ind = b.vocab().stoi[ch]
                                out[r, ind] = max(out[r, ind], out[r, onmt.IO.UNK])
                        nsizes = [1 for _ in new_this_round]
                        for r in range(out.shape[0]):
                            out_scores[new_this_round[r]] += out[r, n] / float(len(self.back_translators))
                        # print('ae')
                        # print(nsizes)
            going_after += time.time() - zaaa

            prev = new_prev
            prev_unks = new_prev_unks
            # print('prev', prev)
            prev_scores = new_prev_scores
            # print("HIHFSD",  prev_scores[2])

            # new_sizes = []
            # idxs = []
            # return []
        if threshold:
            threshold = orig_threshold + orig_score
        if verbose:
            print('first ', feed_original )
            print('between ', in_between)
            print('going after', going_after)
            print('total after', feed_original + in_between + going_after)
        # return [x for x in sorted(out_scores.items(), key=lambda x: x[1], reverse=True)]
        # threshold = -99999999
        return [x for x in sorted(out_scores.items(), key=lambda x: x[1], reverse=True) if x[1] > threshold]
        return []
        key_order = list(out_scores.keys())
        best = -9999999
        for dec_idx, (to, back, mapper, back_mapper, encStates, context, mapping) in enumerate(zip(
                self.to_translators, self.back_translators, self.vocab_mappers,
                self.back_vocab_mappers, encoder_states, contexts, mappings)):
            for i, next_ in zip(after_ids[1:], after_ids[2:]):
                idx = [int(back_mapper[i])]
                n = int(back_mapper[next_])
                for key in key_order:
                    decStates = out_dec_states[key][dec_idx]
                    new_sizes = out_new_sizes[key]
                    out, decStates, attn = back.advance_states(encStates, context,
                                                           decStates, idx, new_sizes)
                    attenz = attn['std'].data[0].cpu().numpy()
                    chosen = np.argmax(attenz, axis=1)
                    for r, ch in enumerate(chosen):
                        ch = mapping[ch]
                        if ch in back.vocab().stoi:
                            # print("YOO")
                            ind = back.vocab().stoi[ch]
                            # print("prev", out[r, ind])
                            out[r, ind] = max(out[r, ind], out[r, onmt.IO.UNK])
                    out_scores[key] += out[0, n]
                    best = max(out[0, n], best)
                    out_dec_states[key][dec_idx] = decStates
            if threshold:
                threshold = best + orig_threshold
                key_order = [k for k, v in out_scores if v > threshold]

        print(sorted(out_scores.items(), key=lambda x: x[1], reverse=True))
        return []

        if run_through:
            orig_ids = np.array([self.global_stoi[x] if x in self.global_stoi else onmt.IO.UNK for x in words_after] +
                                [self.global_stoi[onmt.IO.EOS_WORD]])
            for to, back, mapper, back_mapper, encStates, context, decStates, mapping in zip(
                    self.to_translators, self.back_translators, self.vocab_mappers, self.back_vocab_mappers,
                    enc_states, contexts, dec_states, mappings):
                if not picked.shape[0]:
                    break
                idx = [int(back_mapper[x]) for x in picked]
                # print(idx)
                # print(idx)
                # print([self.global_itos[x] for x in picked])
                # idx = int(back_mapper[x])
                n = int(back_mapper[orig_ids[0]])
                # print(n, back.vocab().itos[n])
                out, decStates, attn = back.advance_states(encStates, context,
                                                           decStates, idx, [len(idx)])

                attenz = attn['std'].data[0].cpu().numpy()
                chosen = np.argmax(attenz, axis=1)
                for r, ch in enumerate(chosen):
                    ch = mapping[ch]
                    if ch in back.vocab().stoi:
                        # print("YOO")
                        ind = back.vocab().stoi[ch]
                        # print("prev", out[r, ind])
                        out[r, ind] = max(out[r, ind], out[r, onmt.IO.UNK])
                # print(n, out[:, n])
                global_scores[picked] += out[:, n]
                sizes = [1 for _ in range(topk)]
                if threshold:
                    to_keep = np.where(global_scores[picked] >= threshold)[0]
                    to_delete = np.where(global_scores[picked] < threshold)[0]
                    picked = picked[to_keep]
                    print(picked.shape)
                    for x in to_delete:
                        sizes[x] = 0
                    topk = picked.shape[0]
                    if not topk:
                        break
                # global_scores[back_mapper] += out[:, n]
                for i, next_ in zip(orig_ids, orig_ids[1:]):
                    idx = [int(back_mapper[i]) for _ in range(topk)]
                    n = int(back_mapper[next_])
                    out, decStates, attn = back.advance_states(encStates, context,
                                                               decStates, idx, sizes)
                    attenz = attn['std'].data[0].cpu().numpy()
                    chosen = np.argmax(attenz, axis=1)
                    for r, ch in enumerate(chosen):
                        ch = mapping[ch]
                        if ch in back.vocab().stoi:
                            # print("YOO")
                            ind = back.vocab().stoi[ch]
                            # print("prev", out[r, ind])
                            out[r, ind] = max(out[r, ind], out[r, onmt.IO.UNK])
                            # print("aft", out[r, ind])
                # print(np.argsort(out[0])[-5:])
                    global_scores[picked] += out[:, n]
                    # print(n, out[:, n])
                    sizes = [1 for _ in range(topk)]
                    if threshold:
                        to_keep = np.where(global_scores[picked] >= threshold)[0]
                        to_delete = np.where(global_scores[picked] < threshold)[0]
                        # print(picked.shape)
                        picked = picked[to_keep]
                        # print(picked.shape)
                        topk = picked.shape[0]
                        for x in to_delete:
                            sizes[x] = 0
                        # print(topk)
                        if not topk:
                            break
        global_scores /= len(self.back_translators)
        last_scores /= len(self.back_translators)
        # TODO: there may be duplicates because of new_Unk
        ret = [(self.global_itos[z], global_scores[z]) if z != onmt.IO.UNK else (new_unk, global_scores[z]) for z in picked if self.global_itos[z] != onmt.IO.EOS_WORD]
        return sorted(ret, key=lambda x: x[1], reverse=True)
        return global_scores, new_unk
        print()
        print(list(reversed([self.global_itos[x] for x in np.argsort(global_scores)[-100:]])))
        pass'''
    def generate_paraphrases(self, sentence, topk=10, threshold=None, edit_distance_cutoff=None, penalize_unks=True, frequent_ngrams=None):
        # returns a list of (sentence, score).
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

        # Always include original sentence in this
        orig = onmt_model.clean_text(sentence)
        output = dict([(orig, orig_score)])
        orig_ids = np.array([self.global_stoi[onmt.IO.BOS_WORD]] + [self.global_stoi[x] if x in self.global_stoi else onmt.IO.UNK for x in orig.split()])
        orig_words = [onmt.IO.BOS_WORD] + orig.split()
        orig_itoi = {}
        orig_stoi = {}
        for i, w in zip(orig_ids, orig_words):
            # if i not in orig_itoi:
            #     idx = len(orig_itoi)
            #     orig_itoi[i] = idx
            if w not in orig_stoi:
                idx = len(orig_stoi)
                orig_stoi[w] = idx
                if i not in orig_itoi:
                    orig_itoi[i] = idx
        # print(sorted([(x, k) for x, k in orig_stoi.items()]))
        # print()
        # print(sorted([(self.global_itos[x], k) for x, k in orig_itoi.items()]))
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
        import time
        while prev:
            orig_time = time.time()
            global_scores = np.zeros((len(prev), (len(self.global_itos))))
            # print(global_scores.shape)
            all_stuff = zip(
                self.back_translators, self.vocab_mappers,
                self.back_vocab_mappers, self.vocab_unks, contexts,
                decoder_states, encoder_states, src_examples, mappings)
            new_decoder_states = []
            new_attns = []
            unk_scores = []
            # print()
            for (b, mapper, back_mapper, unks, context,
                 decStates, encStates, srcz, mapping) in all_stuff:
                idx = [int(back_mapper[i]) for i in idxs]
                out, decStates, attn = b.advance_states(
                    encStates, context, decStates, idx, new_sizes)
                # print(list(reversed([(b.vocab().itos[x], out[0, x]) for x in np.argsort(out[0])[-5:]])))
                new_decoder_states.append(decStates)
                new_attns.append(attn)
                attenz = attn['std'].data[0].cpu().numpy()
                chosen = np.argmax(attenz, axis=1)
                for r, ch in enumerate(chosen):
                    ch = mapping[ch]
                    if ch in b.vocab().stoi:
                        # print("YOO")
                        ind = b.vocab().stoi[ch]
                        # print("prev", out[r, ind])
                        out[r, ind] = max(out[r, ind], out[r, onmt.IO.UNK])
                        # print("aft", out[r, ind])
                    elif ch in self.global_stoi:
                        # print(ch)
                        ind = self.global_stoi[ch]
                        global_scores[r, ind] -= to_add
                        # if ch == 'giraffes':
                        #     print(ind in unks, 30027 in unks, ind ==30027)
                        # break
                # print(list(reversed([(b.vocab().itos[x], out[0, x]) for x in np.argsort(out[0])[-5:]])))
                # print('ya', [mapping[bb] for bb in chosen])
                # print('ya', attenz)
                # print('ya', np.argmax(attenz, axis=1))
                # print(out[:, onmt.IO.UNK])
                # print("AEEE",  30027 in unks)
                unk_scores.append(out[:, onmt.IO.UNK])
                global_scores[:, mapper] += out
                # print(global_scores[:, 30027])
                if unks.shape[0]:
                    # global_scores[:, unks] += out[:, onmt.IO.UNK][:, np.newaxis]
                    global_scores[:, unks] += to_add + out[:, onmt.IO.UNK][:, np.newaxis]
                # print(global_scores[:, 30027])
            # print()
            if PROFILING:
                print(time.time() - orig_time, 'decoding')
                orig_time = time.time()
            decoder_states = new_decoder_states
            global_scores /= float(len(self.back_translators))
            # TODO: Is this right?
            unk_scores = [normalize_ll(x) for x in np.array(unk_scores).T]
            if PROFILING:
                print(time.time() - orig_time, 'normalizing unk scoers')
                orig_time = time.time()
            # print(unk_scores)
            # print(global_scores[0, 0], global_scores[0, 7109], global_scores.max())
            # print(sorted([(self.global_itos[x], global_scores[0, x]) for x in np.argpartition(global_scores[0], -5)[-5:]], key=lambda x:x[1], reverse=True))
            # break
            # for b, back_mapper in zip(self.back_translators, self.back_vocab_mappers):
            #     print([(b.vocab().itos[back_mapper[x]], global_scores[x]) for x in np.argsort(global_scores)[-5:]])

            new_prev = []
            new_prev_distance_rep = []
            new_prev_unks = []
            new_prev_scores = []
            new_sizes = []
            new_origins = []
            idxs = []
            if PROFILING:
                print(time.time() - orig_time, 'before adding scores')
                orig_time = time.time()
            new_scores = global_scores + np.array(prev_scores)[:, np.newaxis]
            if PROFILING:
                print(time.time() - orig_time, 'adding scores')
                orig_time = time.time()
            # print(new_scores.shape)
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
                                i1 = i2# - 1
                                possibles.append([i1 - 1, i1])
                            elif j1 is None:
                                possibles.append([i1])
                            else:
                                possibles.append([i1])
                    if tag == 'delete':
                        for i1 in range(i1, i2):
                            possibles.append([i1])
                if len(possibles) > 1:
                    # print(possibles)
                    possibles.pop(0)
                    # print(possibles)
                return possibles
            if frequent_ngrams is not None:
                for i, p_rep in enumerate(prev_distance_rep):
                    for idx, v in orig_itoi.items():
                        # I'm ignoring UNKs here and letting them be fixed in the next iteration
                        if idx == onmt.IO.UNK:
                            continue
                        candidate = p_rep + [v]
                        # import difflib
                        a = difflib.SequenceMatcher(a = mapped_orig[:len(candidate)], b=candidate)
                        possibles = get_possibles(a.get_opcodes())
                        if len(possibles) == 1 and possibles[0] == tuple():
                            continue
                        if not np.any([x in new_f for x in itertools.product(*possibles)]):
                        # if distance > edit_distance_cutoff:
                            # pass
                            # print (possibles)
                            # print("not allowing", [orig_words[x] if x != -1 else 'unk' for x in candidate])
                            new_scores[i, idx] = -100000
                candidate = p_rep + [-1]
                a = difflib.SequenceMatcher(a = mapped_orig[:len(candidate)], b=candidate)
                possibles = get_possibles(a.get_opcodes())
                if not np.any([x in new_f for x in itertools.product(*possibles)]):
                    new_scores[i, not_in_sentence] = -10000
            if edit_distance_cutoff is not None:
                for i, p_rep in enumerate(prev_distance_rep):
                    for idx, v in orig_itoi.items():
                        # I'm ignoring UNKs here and letting them be fixed in the next iteration
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
            if PROFILING:
                print(time.time() - orig_time, 'edit distance cutoff')
                orig_time = time.time()
            if threshold:
                where = np.where(new_scores > threshold)
                if PROFILING:
                    print(time.time() - orig_time, 'thresholding')
                    orig_time = time.time()
                if topk:
                    # print(where)
                    # print(new_scores[where])
                    # print(threshold)
                    # print(new_scores[where].shape)
                    largest = largest_indices(new_scores[where], topk)[0]
                    where = (where[0][largest], where[1][largest])
                    # print(where)
                    # print(new_scores[where])
                # print(where)
            else:
                # print(new_scores.shape)
                where = largest_indices(new_scores, topk)

            if PROFILING:
                print(time.time() - orig_time, 'topk')
                orig_time = time.time()
            tmp = np.argsort(where[0])
            where = (where[0][tmp], where[1][tmp])
            # TODO: Is this right?
            if (edit_distance_cutoff is not None and
                    len(prev[0]) < len(orig_ids) and
                    orig_ids[len(prev[0])] not in where[1][where[0] == 0]):
                where = (np.hstack(([0], where[0])),
                         np.hstack(([orig_ids[len(prev[0])]], where[1])))
            # print(where[0].shape)
            # print(where)
            # Where needs to be sorted by i, since idxs must be in order of
            # where stuff came from
            for i, j in zip(*where):
                if j == self.global_stoi[onmt.IO.EOS_WORD]:
                    words = [self.global_itos[x] if x != onmt.IO.UNK
                             else prev_unks[i][k]
                             for k, x in enumerate(prev[i][1:], start=1)]
                    new = ' '.join(words)
                    if new not in output:
                        output[new] = new_scores[i, j]
                    else:
                        output[new] = max(output[new], new_scores[i, j])
                    # if topk:
                    #     topk -= 1
                    continue
                new_origins.append(i)
                # print (' '.join(self.global_itos[x] for x in prev[i][1:] + [j]))
                new_unk = '<unk>'
                if j == onmt.IO.UNK:
                    # print(i, j, new_attns[0]['std'].data.shape)
                    new_unk_scores = collections.defaultdict(lambda: 0)
                    for x, src, mapping, score_weight in zip(new_attns, src_examples, mappings, unk_scores[i]):
                        # print(src)
                        attn = x['std'].data[0][i]
                        # TODO: Should we only allow unks here? We are
                        # currently weighting based on the original score, but
                        # this makes it so one always chooses the unk.
                        for zidx, (word, score) in enumerate(zip(src, attn)):
                            # if b.vocab().stoi[word] == onmt.IO.UNK:
                            word = mapping[zidx]
                            new_unk_scores[word] += score * score_weight
                            # print(word, score, score_weight )
                    # print(sorted(new_unk_scores.items(), key=lambda x:x[1], reverse=True))
                    new_unk = max(new_unk_scores.items(),
                                  key=operator.itemgetter(1))[0]
                    # _, max_index = attn.max(0)
                    # # print(new_attns[0]['std'].data[0][i] + new_attns[1]['std'].data[0][i])
                    # max_index = int(max_index[0])
                    # # print(src_examples[0][max_index])
                    # # print(len(src_examples[0]))
                    # new_unk = src_examples[0][max_index]
                # print (' '.join(self.global_itos[x] for x in prev[i][1:]) + ' ' + new_j)
                # new_unk = 'unk'

                if edit_distance_cutoff is not None:
                    distance_rep = orig_itoi[j] if j in orig_itoi else -1
                    if j == onmt.IO.UNK:
                        distance_rep = (orig_stoi[new_unk] if new_unk in orig_stoi
                                        else -1)
                    # print('d', distance_rep)
                    new_prev_distance_rep.append(prev_distance_rep[i] +
                                                 [distance_rep])
                new_prev.append(prev[i] + [j])
                new_prev_unks.append(prev_unks[i] + [new_unk])
                new_prev_scores.append(new_scores[i, j])
                idxs.append(j)
            # print(idxs)

            # print(mapped_orig[:len(prev[0])+ 1])
            # print(new_prev_distance_rep)
            if PROFILING:
                print(time.time() - orig_time, 'processing where')
                orig_time = time.time()
            new_sizes = np.bincount(new_origins, minlength=len(prev))
            new_sizes = [int(x) for x in new_sizes]
            prev = new_prev
            prev_unks = new_prev_unks
            prev_distance_rep = new_prev_distance_rep
            # print('prev', prev)
            prev_scores = new_prev_scores
            if topk and len(output) > topk:
                break
            # for z, unks, dr in zip(prev, prev_unks, prev_distance_rep):
            for z, s, unks in zip(prev, prev_scores, prev_unks):
            #     import editdistance
            #     # TODO: Must ignore unks here - it's fine if I think it's an unk if the text is the same
            #     z = np.array(z)
            #     non_unk = np.where(z != onmt.IO.UNK)[0]
            #     # print(list(zip(z[non_unk], orig_ids[non_unk])))
            #     distance = editdistance.eval(z[non_unk], orig_ids[non_unk])
            #     bla = list(zip(*([(unks[i], orig_words[i]) for i in range(len(z)) if z[i] == onmt.IO.UNK])))
            #     unk_distance = 0
            #     if bla:
            #         unks1, unks2 = bla
            #         unk_distance = editdistance.eval(unks1, unks2)
                words = [self.global_itos[x] if x != onmt.IO.UNK else 'UNK'+ unks[k] for k, x in enumerate(z)]
                # print(words, s, prev_unks)
            #     d2 = editdistance.eval(dr, mapped_orig[:len(dr)])
            #     # print(words, distance, unk_distance, 'd2', d2, dr, mapped_orig[:len(dr)])
            # print()
            if PROFILING:
                print(time.time() - orig_time, 'rest')
                orig_time = time.time()

        return sorted(output.items(), key=lambda x:x[1], reverse=True)
        # print 'generate_paraphrases', n
        # all_generated = []
        # for to, back in zip(self.to_translators, self.back_translators):
        #     translation = choose_forward_translation(sentence, to, back, n=5)
        #     all_generated.extend(back.translate([translation], n_best=n)[0])
        # all_generated = list(set([x[0].encode('ascii', 'ignore').decode() for x in all_generated if x[0]]))
        # scores = self.score_sentences(sentence, all_generated)
        # return sorted(zip(all_generated, scores), key=lambda x: x[1],
        #               reverse=True)

    def test_translators(self, sentence):
        print('original:', sentence)
        print()
        for to, back in zip(self.to_translators, self.back_translators):
            # translation = to.translate([sentence], n_best=1)[0][0][0]
            translation, mapping = choose_forward_translation(sentence, to, back, n=5)
            print(translation)
            b = back.translate([translation], n_best=1)[0][0]
            print(b)
            print(sentence)
            print('score_original:', back.score(translation, [sentence]))
            print()

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
    def score_sentences_old(self, original_sentence, other_sentences,
                        relative_to_original=False):
        # returns a numpy array of scores, one for each sentence in
        # other_sentences
        memoized_stuff = self.last == original_sentence
        if not memoized_stuff:
            self.last = original_sentence
            self.memoized = {}
            self.memoized['translation'] = []
        all_scores = []
        if relative_to_original:
            other_sentences = [original_sentence] + other_sentences
        for k, (to, back) in enumerate(zip(self.to_translators, self.back_translators)):
            if memoized_stuff:
                translation, mapping = self.memoized['translation'][k]
            else:
                translation, mapping = choose_forward_translation(original_sentence, to, back,
                                                         n=5)
                self.memoized['translation'].append((translation, mapping))
            # if I want to pivot over multiple translations, this is how to do it:
            # trs = to.translate([original_sentence], n_best=5)[0]
            # translations = [x[0] for x in trs]
            # weights = np.array([x[1] for x in trs])
            # # normalizing using exp-normalize trick
            # weights = np.exp(weights - weights.max())
            # weights = weights / weights.sum()
            # # print 'w',weights
            # temp_scores = []
            # for t in translations:
            #     scores = back.score(t, other_sentences)
            #     temp_scores.append(scores)
            # scores = np.average(temp_scores, axis=0, weights=weights)

            scores = back.score(translation, other_sentences)
            all_scores.append(scores)
        scores = np.mean(all_scores, axis=0)
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
