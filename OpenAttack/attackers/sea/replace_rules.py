import difflib
import itertools
import numpy as np
import re
import enum
import collections
import copy
import sys
from itertools import zip_longest as zip_longest
unicode = lambda x: x


def clean_text(text, only_upper=False):
    text = '%s%s' % (text[0].upper(), text[1:])
    if only_upper:
        return text
    text = text.replace('|', 'UNK')
    text = re.sub('(^|\s)-($|\s)', r'\1@-@\2', text)
    text = re.sub(' (n)(\'.) ', r'\1 \2 ', text)
    return text


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    if n > flat.shape[0]:
        indices = np.array(range(flat.shape[0]), dtype='int')
        return np.unravel_index(indices, ary.shape)
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

class OpToken:
    def __init__(self, type_, value):
        self.type = type_
        self.value = value

    def test(self, token):
        if self.type == 'text':
            return token.text == self.value
        if self.type == 'pos':
            return token.pos == self.value
        if self.type == 'tag':
            return token.tag == self.value

    def hash(self):
        return self.type + '_' + self.value

Token = collections.namedtuple('Token', ['text', 'pos', 'tag'])

def capitalize(text):
    if len(text) == 0:
        return text
    if len(text) == 1:
        return text.upper()
    else:
        return '%s%s' % (text[0].upper(), text[1:])

class Tokenizer:
    def __init__(self, nlp):
        self.nlp = nlp

    def tokenize(self, texts):
        ret = []
        for text in texts:
            text_token =  self.nlp.get_tokens(text)
            token_sequence = [Token(x[0], x[1], x[1]) for x in text_token]
            ret.append(token_sequence)
        return ret

    def tokenize_text(self, texts):
        ret = [list(map(lambda x: x[0], self.nlp.get_tokens(text))) for text in texts]
        return [" ".join(r) for r in ret]

    def clean_for_model(self, texts):
        fn = lambda x: re.sub(r'\s+', ' ', re.sub(r'\s\'(\w{1, 3})', r"'\1", x).replace('@-@', '-').strip())
        return self.tokenize_text([fn(capitalize(x)) for x in texts])

    def clean_for_humans(self, texts):
        return [re.sub("\s(n')", r'\1', re.sub(r'\s\'(\w)', r"'\1", capitalize(x))) for x in texts]




class ReplaceRule:
    def __init__(self, op_sequence, replace_sequence):
        self.op_sequence = op_sequence
        self.replace_sequence = replace_sequence

    def apply(self, token_sequence, status_only=False, return_position=False, fix_apostrophe=True):
        token_sequence = [Token('<s>', '<s>', '<s>')] + token_sequence + [Token('</s>', '</s>', '</s>')]
        match_idx = 0
        size_seq = len(self.op_sequence)
        matched = -1
        matched_pos = collections.defaultdict(lambda: [])
        for i, t in enumerate(token_sequence):
            if self.op_sequence[match_idx].test(t):
                if self.op_sequence[match_idx].type == 'pos':
                    matched_pos[t.pos].append(token_sequence[i].text)
                if self.op_sequence[match_idx].type == 'tag':
                    matched_pos[t.tag].append(token_sequence[i].text)
                match_idx += 1
                if match_idx == size_seq:
                    matched = i
                    break
            else:
                match_idx = 0
                matched_pos = collections.defaultdict(lambda: [])
        status = matched > 0
        if status_only:
            return status
        if not status:
            if return_position:
                return status, '', -1
            return status, ''
        match_start = matched - size_seq + 1
        t_before = [x.text for x in token_sequence[1:match_start]]
        t_after = [x.text for x in token_sequence[matched + 1:-1]]
        t_mid = []
        for x in self.replace_sequence:
            if x.type == 'text':
                t_mid.append(x.value)
            else:
                text = matched_pos[x.value].pop(0)
                t_mid.append(text)

        ret_text = " ".join(t_before + t_mid + t_after)
        if fix_apostrophe:
            ret_text = ret_text.replace(' \'', '\'')
        if return_position:
            return True, ret_text, match_start - 1
        return True, ret_text

    def apply_to_texts(self, token_sequences, idxs_only=False, fix_apostrophe=True):
        idxs = []
        new_texts = []
        for i, token_seq in enumerate(token_sequences):
            status, ntext = self.apply(token_seq, status_only=idxs_only, fix_apostrophe=fix_apostrophe)
            if status:
                idxs.append(i)
                new_texts.append(ntext)
        return np.array(idxs), new_texts

    def hash(self):
        return " ".join([op.hash() for op in self.op_sequence]) + ' -> ' + " ".join([op.hash() for op in self.replace_sequence])

class TextToReplaceRules:
    def __init__(self, nlp, from_dataset, flip_dataset=[], min_freq=.01, min_flip=0.01, ngram_size=4):
        if len(flip_dataset) != 0:
            assert len(from_dataset) == len(flip_dataset)
        self.tokenizer = Tokenizer(nlp)
        self.min_freq = min_freq * len(from_dataset)
        self.min_flip = min_flip * len(flip_dataset)
        self.ngram_size = ngram_size

        self.ngram_freq = collections.defaultdict(lambda: 0.)
        token_sequences = self.tokenizer.tokenize(from_dataset)
        ngram_idxs = collections.defaultdict(lambda: [])
        for i, s in enumerate(token_sequences):
            positions = self.get_positions(s, ngram_size)
            for p in positions:
                self.ngram_freq[p] += 1
                ngram_idxs[p].append(i)
        all_ngrams = list(self.ngram_freq.keys())
        self.ngram_idxs = {}
        for ngram in all_ngrams:
            self.ngram_idxs[ngram] = set(ngram_idxs[ngram])

        self.ngram_flip_freq = collections.defaultdict(lambda: 0.)
        for i, others in enumerate(flip_dataset):
            if i % 1000 == 0:
                print(i)
            token_sequences = self.tokenizer.tokenize(others)
            ngrams_flipped = set()
            for s in token_sequences:
                positions = self.get_positions(s, ngram_size)
                for p in positions:
                    ngrams_flipped.add(p)
            for n in ngrams_flipped:
                self.ngram_flip_freq[n] += 1




    def is_param_ngram_frequent(self, ngram, flip=False):
        if type(ngram) != list:
            ngram = tuple([ngram.hash()])
        else:
            ngram = tuple([x.hash() for x in ngram])

        if flip:
            return self.ngram_flip_freq[ngram] >= self.min_flip
        else:
            return self.ngram_freq[ngram] >= self.min_freq

    def get_rule_idxs(self, rule):
        ngram = rule.op_sequence
        ngram = tuple([x.hash() for x in ngram])
        return self.ngram_idxs[ngram]


    def get_positions(self, tokenized_sentence, ngram_size):
        def get_params(token):
            return (OpToken('text', token.text),
                    OpToken('pos', token.pos),
                    OpToken('tag', token.tag))

        positions = {}
        prev = Token('<s>', '<s>', '<s>')
        for i, current in enumerate(tokenized_sentence + [Token('</s>', '</s>', '</s>')]):
            for j in range(0, ngram_size):
                if i - j < 0:
                    continue
                to_consider = tokenized_sentence[i - j:i + 1]
                tokens = [get_params(x) for x in to_consider]
                ngrams = [tuple([y.hash() for y in x]) for x in itertools.product(*tokens)]
                for ngram in ngrams:
                    if ngram == tuple():
                        continue
                    positions.setdefault(ngram, i)
        return positions


    def compute_rules(self, sentence, others, use_words=True, use_pos=True, use_tags=False, max_rule_length=3):

        def get_params(token):
            to_ret = [OpToken('text', token.text)]
            if use_pos:
                to_ret.append(OpToken('pos', token.pos))
            if use_tags:
                to_ret.append(OpToken('tag', token.tag))
            return tuple(to_ret)

        doc = self.tokenizer.tokenize([unicode(sentence)])[0]
        other_docs = self.tokenizer.tokenize([unicode(x) for x in others])


        sentence = [x.text for x in doc]

        others = [[x.text for x in d] for d in other_docs]


        all_rules = []
        n_doc = [Token('<s>', '<s>', '<s>')] + doc + [Token('</s>', '</s>', '</s>')]
        for other, other_doc in zip(others, other_docs):
            n_other = [Token('<s>', '<s>', '<s>')] + other_doc + [Token('</s>', '</s>', '</s>')]
            matcher = difflib.SequenceMatcher(a=sentence, b=other)
            ops = ([x for x in matcher.get_opcodes() if x[0] != 'equal'])
            if len(ops) == 0:
                all_rules.append([])
                continue
            reps = [n_doc[op[1]:op[2]] for op in ops]
            withs = [n_other[op[3]:op[4]] for op in ops]

            def check_pos(ops1, ops2):
                counter = collections.Counter()
                for o in ops2:
                    if o.type != 'text':
                        counter[o.value] += 1
                for o in ops1:
                    if o.type != 'text':
                        counter[o.value] -= 1
                most_common = counter.most_common(1)
                if len(most_common) == 0 or most_common[0][1] <= 0:
                    return True
                return False



            this_rules = []
            other_sentence = " ".join(other)
            for rep, withe in zip(reps, withs):
                if len(rep) > self.ngram_size or len(rep) == 0:
                    continue
                
                tokens = [get_params(x) for x in rep]
                
                ngrams = [[y for y in x] for x in itertools.product(*tokens)]
                tokens_o = [get_params(x) if x in rep else (OpToken('text', x.text),) for x in withe]
                ngrams_o = [[y for y in x] for x in itertools.product(*tokens_o)]

                frequent = [x for x in ngrams if self.is_param_ngram_frequent(x)]
                

                frequent_other = [x for x in ngrams_o if self.is_param_ngram_frequent(x, flip=True) or x == []]
                rules = [ReplaceRule(a, b) for a, b in (itertools.product(frequent, frequent_other)) if check_pos(a, b)]

                ngrams = [[y for y in x] for x in itertools.product(*tokens)]
                this_rules.extend(rules)

            all_rules.append(this_rules)
        return all_rules
