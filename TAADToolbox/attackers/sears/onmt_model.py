import argparse
import torch
import onmt
import numpy as np
import re
import sys
import torchtext

from collections import Counter, defaultdict


def repeat(repeat_numbers, tensor):
    cat = []
    for i, x in enumerate(repeat_numbers):
        if x == 0:
            continue
        cat.append(tensor[:, i:i+1, :].repeat(1, x, 1))
    return torch.cat(cat, 1)


def transform_dec_states(decStates, repeat_numbers):
    assert len(repeat_numbers) == decStates._all[0].data.shape[1]
    with torch.no_grad():
        vars = [repeat(repeat_numbers, e.data) for e in decStates._all]
    decStates.hidden = tuple(vars[:-1])
    decStates.input_feed = vars[-1]

def clean_text(text, only_upper=False):
    # should there be a str here?`
    text = '%s%s' % (text[0].upper(), text[1:])
    if only_upper:
        return text
    text = text.replace('|', 'UNK')
    text = re.sub('(^|\s)-($|\s)', r'\1@-@\2', text)
    # text = re.sub(' (n?\'.) ', r'\1 ', text)
    # fix apostrophe stuff according to tokenizer
    text = re.sub(' (n)(\'.) ', r'\1 \2 ', text)
    text = text.replace('<br />', ' ')
    return text

class OnmtModel(object):
    def __init__(self, model_path, gpu_id=1):
        parser = argparse.ArgumentParser(description='translate.py')
        parser.add_argument('-model', required=True,
                            help='Path to model .pt file')
        parser.add_argument(
            '-src', required=True,
            help='Source sequence to decode (one line per sequence)')
        parser.add_argument('-src_img_dir',   default="",
                            help='Source image directory')
        parser.add_argument('-tgt',
                            help='True target sequence (optional)')
        parser.add_argument('-output', default='pred.txt',
                            help="""Path to output the predictions (each line will
                            be the decoded sequence""")
        parser.add_argument('-beam_size',  type=int, default=5,
                            help='Beam size')
        parser.add_argument('-batch_size', type=int, default=30,
                            help='Batch size')
        parser.add_argument('-max_sent_length', type=int, default=100,
                            help='Maximum sentence length.')
        parser.add_argument('-replace_unk', action="store_true",
                            help="""Replace the generated UNK tokens with the
                            source token that had highest attention weight. If
                            phrase_table is provided, it will lookup the
                            identified source token and give the corresponding
                            target token. If it is not provided (or the
                            identified source token does not exist in the
                            table) then it will copy the source token""")
        parser.add_argument(
            '-verbose', action="store_true",
            help='Print scores and predictions for each sentence')
        parser.add_argument('-attn_debug', action="store_true",
                            help='Print best attn for each word')
        parser.add_argument('-dump_beam', type=str, default="",
                            help='File to dump beam information to.')

        parser.add_argument('-n_best', type=int, default=1,
                            help="""If verbose is set, will output the n_best
                            decoded sentences""")

        parser.add_argument('-gpu', type=int, default=-1,
                            help="Device to run on")
        # options most relevant to summarization
        parser.add_argument('-dynamic_dict', action='store_true',
                            help="Create dynamic dictionaries")
        parser.add_argument('-share_vocab', action='store_true',
                            help="Share source and target vocabulary")
        # Alpha and Beta values for Google Length + Coverage penalty
        # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
        parser.add_argument('-alpha', type=float, default=0.0,
                            help="""Google NMT length penalty parameter
                            (higher = longer generation)""")
        parser.add_argument('-beta', type=float, default=0.0,
                            help="""Coverage penalty parameter""")

        opt = parser.parse_args(( '-model %s -src /tmp/a -tgt /tmp/b -output /tmp/c -gpu %d -verbose -beam_size 5 -batch_size 1 -n_best 5 -replace_unk' % (model_path, gpu_id)).split()) # noqa
        opt.cuda = opt.gpu > -1
        if opt.cuda:
            torch.cuda.set_device(opt.gpu)
        self.translator = onmt.Translator(opt)


    def get_init_states(self, sentence):
        sentence = clean_text(sentence)
        data = ONMTDataset2([sentence], None, self.translator.fields,
                            None)
        opt = self.translator.opt
        self.translator.opt.tgt = None
        testData = onmt.IO.OrderedIterator(
            dataset=data, device=opt.gpu,
            batch_size=opt.batch_size, train=False, sort=False,
            shuffle=False)
        batch = next(testData.__iter__())
        _, src_lengths = batch.src
        src = onmt.IO.make_features(batch, 'src')
        encStates, context = self.translator.model.encoder(src, src_lengths)
        decStates = self.translator.model.decoder.init_decoder_state(
                                        src, context, encStates)
        src_example = batch.dataset.examples[batch.indices[0].data.item()].src
        return encStates, context, decStates, src_example

    def advance_states(self, encStates, context, decStates, new_idxs,
                       new_sizes):
        # new_idxs is a list of new inputs
        # new_sizes indicates how duplicates to make of each decStates in the
        #   previous round
        # Returns predict_proba, decStates(updated)
        tt = torch.cuda if self.translator.opt.cuda else torch


        def var(a): 
            with torch.no_grad():
                return a

        def rvar(a, l): return var(a.repeat(1, l, 1))
        current_state = tt.LongTensor(new_idxs)
        inp = var(torch.stack([current_state]).t().contiguous().view(1, -1))
        inp = inp.unsqueeze(2)
        n_context = rvar(context.data, len(new_idxs))
        transform_dec_states(decStates, new_sizes)
        decOut, decStates, attn = self.translator.model.decoder(inp, n_context,
                                                                decStates)
        decOut = decOut.squeeze(0)
        out = self.translator.model.generator.forward(decOut).data
        out_np = out.cpu().numpy()
        return out_np, decStates, attn


    def vocab(self):
        return self.translator.fields['tgt'].vocab

    def translate(self, sentences, n_best=1, return_from_mapping=False):
        # Returns a 2d list (len(sentences), n(best)) of pairs, where each
        # is a translation and a score
        sentences = [clean_text(x) for x in sentences]
        data = ONMTDataset2(sentences, None, self.translator.fields,
                            None)
        opt = self.translator.opt
        self.translator.opt.tgt = None
        testData = onmt.IO.OrderedIterator(
            dataset=data, device=opt.gpu,
            batch_size=opt.batch_size, train=False, sort=False,
            shuffle=False)
        out = []
        scores = []
        mappings = []
        # gold = []
        self.translator.opt.n_best = n_best
        prev_beam_size = self.translator.opt.beam_size
        vocab = self.translator.fields['tgt'].vocab
        if n_best > self.translator.opt.beam_size:
            self.translator.opt.beam_size = n_best
        for batch in testData:
            _, lens = batch.src
            # This only works if batch_size is one

            predBatch, goldBatch, predScore, goldScore, attn, src = (
                self.translator.translate(batch, data))
            # This is doing replace_unk
            if self.translator.opt.replace_unk:
                src_example = batch.dataset.examples[batch.indices[0].data.item()].src
                for i, x in enumerate(predBatch):
                    for j, sentence in enumerate(x):
                        for k, word in enumerate(sentence):
                            if word == vocab.itos[onmt.IO.UNK]:
                                _, maxIndex = attn[i][j][k].max(0)
                                m = int(maxIndex.item())
                                predBatch[i][j][k] = src_example[m]
                                # print 'ae', word, src_example[m]
            if return_from_mapping:
                this_mappings = []
                src_example = batch.dataset.examples[batch.indices[0].data.item()].src
                for i, x in enumerate(predBatch):
                    for j, sentence in enumerate(x):
                        mapping = {}
                        for k, word in enumerate(sentence):
                            _, maxIndex = attn[i][j][k].max(0)
                            m = int(maxIndex.item())
                            mapping[k] = src_example[m]
                        this_mappings.append(mapping)

                mappings.append(this_mappings)
            out.extend([[' '.join(x) for x in y] for y in predBatch])
    #         print predScore
    #         print goldScore
            scores.extend([x[:self.translator.opt.n_best] for x in predScore])
            # gold.extend([x for x in goldScore])
        self.translator.opt.beam_size = prev_beam_size
        if return_from_mapping:
            return [list(zip(x, y, z)) for x, y, z in zip(out, scores, mappings)]
        return [list(zip(x, y)) for x, y in zip(out, scores)]

    def score(self, original_sentence, other_sentences):
        original_sentence = clean_text(original_sentence)
        other_sentences = [clean_text(x) for x in other_sentences]
        # print(original_sentence, other_sentences)
        # print other_sentences
        sentences = [original_sentence] * len(other_sentences)
        self.translator.opt.tgt = 'yes'
        data = ONMTDataset2(sentences, other_sentences, self.translator.fields,
                            None)
        opt = self.translator.opt
        testData = onmt.IO.OrderedIterator(
            dataset=data, device=opt.gpu,
            batch_size=opt.batch_size, train=False, sort=False,
            shuffle=False)
        gold = []
        # print(original_sentence, other_sentences)
        for batch in testData:
            # print('a')
            scores = self.translator._runTarget(batch, data)
            gold.extend([x for x in scores.cpu().numpy()[0]])
        return np.array(gold)


def extractFeatures(tokens):
    "Given a list of token separate out words and features (if any)."
    words = []
    features = []
    numFeatures = None

    for t in range(len(tokens)):
        field = tokens[t].split(u"|")
        word = field[0]
        if len(word) > 0:
            words.append(word)
            if numFeatures is None:
                numFeatures = len(field) - 1
            else:
                assert (len(field) - 1 == numFeatures), \
                    "all words must have the same number of features"

            if len(field) > 1:
                for i in range(1, len(field)):
                    if len(features) <= i-1:
                        features.append([])
                    features[i - 1].append(field[i])
                    assert (len(features[i - 1]) == len(words))
    return words, features, numFeatures if numFeatures else 0


class ONMTDataset2(torchtext.data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        "Sort in reverse size order"
        return -len(ex.src)

    def __init__(self, src_path, tgt_path, fields, opt,
                 src_img_dir=None, **kwargs):
        "Create a TranslationDataset given paths and fields."
        if src_img_dir:
            self.type_ = "img"
        else:
            self.type_ = "text"

        examples = []
        src_words = []
        self.src_vocabs = []
        for i, src_line in enumerate(src_path):
            src_line = src_line.split()
            # if len(src_line) == 0:
            #     skip[i] = True
            #     continue
            if self.type_ == "text":
                # Check truncation condition.
                if opt is not None and opt.src_seq_length_trunc != 0:
                    src_line = src_line[:opt.src_seq_length_trunc]
                src, src_feats, _ = extractFeatures(src_line)
                d = {"src": src, "indices": i}
                self.nfeatures = len(src_feats)
                for j, v in enumerate(src_feats):
                    d["src_feat_"+str(j)] = v
                examples.append(d)
                src_words.append(src)

                # Create dynamic dictionaries
                if opt is None or opt.dynamic_dict:
                    # a temp vocab of a single source example
                    src_vocab = torchtext.vocab.Vocab(Counter(src))

                    # mapping source tokens to indices in the dynamic dict
                    src_map = torch.LongTensor(len(src)).fill_(0)
                    for j, w in enumerate(src):
                        src_map[j] = src_vocab.stoi[w]

                    self.src_vocabs.append(src_vocab)
                    examples[i]["src_map"] = src_map

        if tgt_path is not None:
            for i, tgt_line in enumerate(tgt_path):
                # if i in skip:
                #     continue
                tgt_line = tgt_line.split()

                # Check truncation condition.
                if opt is not None and opt.tgt_seq_length_trunc != 0:
                    tgt_line = tgt_line[:opt.tgt_seq_length_trunc]

                tgt, _, _ = extractFeatures(tgt_line)
                examples[i]["tgt"] = tgt

                if opt is None or opt.dynamic_dict:
                    src_vocab = self.src_vocabs[i]
                    # Map target tokens to indices in the dynamic dict
                    mask = torch.LongTensor(len(tgt)+2).fill_(0)
                    for j in range(len(tgt)):
                        mask[j+1] = src_vocab.stoi[tgt[j]]
                    examples[i]["alignment"] = mask
            assert i + 1 == len(examples), "Len src and tgt do not match"
        keys = examples[0].keys()
        fields = [(k, fields[k]) for k in keys]
        examples = list([torchtext.data.Example.fromlist([ex[k] for k in keys],
                                                         fields)
                         for ex in examples])

        def filter_pred(example):
            return 0 < len(example.src) <= opt.src_seq_length \
                and 0 < len(example.tgt) <= opt.tgt_seq_length

        super(ONMTDataset2, self).__init__(examples, fields,
                                           filter_pred if opt is not None
                                           else None)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(ONMTDataset2, self).__reduce_ex__()

    def collapseCopyScores(self, scores, batch, tgt_vocab):
        """Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        offset = len(tgt_vocab)
        for b in range(batch.batch_size):
            index = batch.indices.data[b]
            src_vocab = self.src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    scores[:, b, ti] += scores[:, b, offset + i]
                    scores[:, b, offset + i].fill_(1e-20)
        return scores

    @staticmethod
    def load_fields(vocab):
        vocab = dict(vocab)
        fields = ONMTDataset2.get_fields(
            len(ONMTDataset2.collect_features(vocab)))
        for k, v in vocab.items():
            # Hack. Can't pickle defaultdict :(
            v.stoi = defaultdict(lambda: 0, v.stoi)
            fields[k].vocab = v
        return fields

    @staticmethod
    def save_vocab(fields):
        vocab = []
        for k, f in fields.items():
            if 'vocab' in f.__dict__:
                f.vocab.stoi = dict(f.vocab.stoi)
                vocab.append((k, f.vocab))
        return vocab

    @staticmethod
    def collect_features(fields):
        feats = []
        j = 0
        while True:
            key = "src_feat_" + str(j)
            if key not in fields:
                break
            feats.append(key)
            j += 1
        return feats

    @staticmethod
    def get_fields(nFeatures=0):
        fields = {}
        fields["src"] = torchtext.data.Field(
            pad_token=onmt.IO.PAD_WORD,
            include_lengths=True)

        # fields = [("src_img", torchtext.data.Field(
        #     include_lengths=True))]

        for j in range(nFeatures):
            fields["src_feat_"+str(j)] = \
                torchtext.data.Field(pad_token=onmt.IO.PAD_WORD)

        fields["tgt"] = torchtext.data.Field(
            init_token=onmt.IO.BOS_WORD, eos_token=onmt.IO.EOS_WORD,
            pad_token=onmt.IO.PAD_WORD)

        def make_src(data, _):
            src_size = max([t.size(0) for t in data])
            src_vocab_size = max([t.max() for t in data]) + 1
            alignment = torch.FloatTensor(src_size, len(data),
                                          src_vocab_size).fill_(0)
            for i in range(len(data)):
                for j, t in enumerate(data[i]):
                    alignment[j, i, t] = 1
            return alignment

        fields["src_map"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.FloatTensor,
            postprocessing=make_src, sequential=False)

        def make_tgt(data, _):
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.LongTensor(tgt_size, len(data)).fill_(0)
            for i in range(len(data)):
                alignment[:data[i].size(0), i] = data[i]
            return alignment

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            postprocessing=make_tgt, sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            sequential=False)

        return fields

    @staticmethod
    def build_vocab(train, opt):
        fields = train.fields
        fields["src"].build_vocab(train, max_size=opt.src_vocab_size,
                                  min_freq=opt.src_words_min_frequency)
        for j in range(train.nfeatures):
            fields["src_feat_" + str(j)].build_vocab(train)
        fields["tgt"].build_vocab(train, max_size=opt.tgt_vocab_size,
                                  min_freq=opt.tgt_words_min_frequency)

        # Merge the input and output vocabularies.
        if opt.share_vocab:
            # `tgt_vocab_size` is ignored when sharing vocabularies
            merged_vocab = onmt.IO.merge_vocabs(
                [fields["src"].vocab, fields["tgt"].vocab],
                vocab_size=opt.src_vocab_size)
            fields["src"].vocab = merged_vocab
            fields["tgt"].vocab = merged_vocab
