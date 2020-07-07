import os
import pickle
import numpy as np
import collections
from ...utils import check_parameters
from ...text_processors import DefaultTextProcessor
from ...attacker import Attacker
from ...data_manager import DataManager

DEFAULT_TO_PATHS = ['english_french_model_acc_71.05_ppl_3.71_e13.pt', 'translation_models/english_portuguese_model_acc_70.75_ppl_4.32_e13.pt']
DEFAULT_BACK_PATHS = ['french_english_model_acc_68.51_ppl_4.43_e13.pt', 'translation_models/portuguese_english_model_acc_69.93_ppl_5.04_e13.pt']

DEFAULT_CONFIG = {
    "processor": DefaultTextProcessor(),
    "TO_PATHS": DEFAULT_TO_PATHS,
    "BACK_PATHS": DEFAULT_BACK_PATHS,
}

class SEARSAttacker(Attacker):
    def __init__(self, **kwargs):
        self.paraphrase_scorer = __import__("paraphrase_scorer", globals={
            "__name__":__name__,
            "__package__": __package__,
        }, level=1)
        self.onmt_model = __import__("onmt_model", globals={
            "__name__":__name__,
            "__package__": __package__,
        }, level=1)
        self.replace_rules = __import__("replace_rules", globals={
            "__name__":__name__,
            "__package__": __package__,
        }, level=1)
        self.rule_picking = __import__("rule_picking", globals={
            "__name__":__name__,
            "__package__": __package__,
        }, level=1)
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        check_parameters(DEFAULT_CONFIG, self.config)
        self.ps = self.paraphrase_scorer.ParaphraseScorer(gpu_id=0)
        self.tokenizer = self.replace_rules.Tokenizer(self.config["processor"])

    def __call__(self, clsf, x_orig, target=None):
        x_orig = x_orig.lower()
        if target is None:
            targeted = False
            target = clsf.get_pred([x_orig])[0]  # calc x_orig's prediction
        else:
            targeted = True
        val_for_onmt = [self.onmt_model.clean_text(x_orig, only_upper=False)]
        orig_scores = {}
        flips = collections.defaultdict(lambda: [])
        right_val = [x_orig]
        right_preds = clsf.get_pred(right_val)
        fs = self.find_flips(x_orig, clsf, topk=100, threshold=-10)
        flips[x_orig].extend([x[0] for x in fs])

        tr2 = self.replace_rules.TextToReplaceRules(self.config["processor"], right_val, [], min_freq=0.005, min_flip=0.00, ngram_size=4)
        frequent_rules = []
        rule_idx = {}
        rule_flips = {}
        for z, f in enumerate(flips):
            rules = tr2.compute_rules(f, flips[f], use_pos=True, use_tags=True)
            for rs in rules:
                for r in rs:
                    if r.hash() not in rule_idx:
                        i = len(rule_idx)
                        rule_idx[r.hash()] = i
                        rule_flips[i] = []
                        frequent_rules.append(r)
                    i = rule_idx[r.hash()]
                    rule_flips[i].append(z)

        token_right = self.tokenizer.tokenize(right_val)
        model_preds = {}

        rule_flips = {}
        rule_other_texts = {}
        rule_other_flips = {}
        rule_applies = {}
        for i, r in enumerate(frequent_rules):
            idxs = list(tr2.get_rule_idxs(r))
            to_apply = [token_right[x] for x in idxs]
            applies, nt = r.apply_to_texts(to_apply, fix_apostrophe=False)
            applies = [idxs[x] for x in applies]
            old_texts = [right_val[i] for i in applies]
            old_labels = right_preds[applies]
            to_compute = [x for x in nt if x not in model_preds]
            if to_compute:
                preds = clsf.get_pred(to_compute)
                for x, y in zip(to_compute, preds):
                    model_preds[x] = y
            new_labels = np.array([model_preds[x] for x in nt])
            where_flipped = np.where(new_labels != old_labels)[0]
            flips = sorted([applies[x] for x in where_flipped])
            rule_flips[i] = flips
            rule_other_texts[i] = nt
            rule_other_flips[i] = where_flipped
            rule_applies[i] = applies
        really_frequent_rules = [i for i in range(len(rule_flips)) if len(rule_flips[i]) > 1]
        threshold = -7.15
        orig_scores = {}
        for i, t in enumerate(right_val):
            orig_scores[i] = self.ps.score_sentences(t, [t])[0]
        ps_scores = {}
        self.ps.last = None

        rule_scores = []
        rejected = set()
        for idx, i in enumerate(really_frequent_rules):
            orig_texts =  [right_val[z] for z in rule_applies[i]]
            orig_scor = [orig_scores[z] for z in rule_applies[i]]
            scores = np.ones(len(orig_texts)) * -50

            decile = np.ceil(.1 * len(orig_texts))
            new_texts = rule_other_texts[i]
            bad_scores = 0
            for j, (o, n, orig) in enumerate(zip(orig_texts, new_texts, orig_scor)):
                if o not in ps_scores:
                    ps_scores[o] = {}
                if n not in ps_scores[o]:
                    if n == '':
                        score = -40
                    else:
                        score = self.ps.score_sentences(o, [n])[0]
                    ps_scores[o][n] = min(0, score - orig)
                scores[j] = ps_scores[o][n]
                if ps_scores[o][n] < threshold:
                    bad_scores += 1
                if bad_scores >= decile:
                    rejected.add(idx)
                    break
            rule_scores.append(scores)
        rule_flip_scores = [rule_scores[i][rule_other_flips[really_frequent_rules[i]]] for i in range(len(rule_scores))]
        frequent_flips = [np.array(rule_applies[i])[rule_other_flips[i]] for i in really_frequent_rules]
        rule_precsupports = [len(rule_applies[i]) for i in really_frequent_rules]

        threshold=-7.15
        disqualified = self.rule_picking.disqualify_rules(rule_scores, frequent_flips,
                          rule_precsupports, 
                      min_precision=0.0, min_flips=6, 
                         min_bad_score=threshold, max_bad_proportion=.10,
                          max_bad_sum=999999)

        x = self.rule_picking.choose_rules_coverage(rule_flip_scores, frequent_flips, None,
                          None, len(right_preds),
                                frequent_scores_on_all=None, k=10, metric='max',
                      min_precision=0.0, min_flips=0, exp=True,
                         min_bad_score=threshold, max_bad_proportion=.1,
                          max_bad_sum=999999,
                         disqualified=disqualified,
                         start_from=[])
        
        for r in x:
            rid = really_frequent_rules[r]
            rule =  frequent_rules[rid]
            for f in rule_flips[rid][:2]:
                new = rule.apply(token_right[f])[1]
                ans = clsf.get_pred([new])[0]
                if targeted:
                    if ans == target:
                        return new, ans
                else:
                    if ans != target:
                        return new, ans
        return None



    def find_flips(self, instance, clsf, topk=10, threshold=-10):
        orig_pred = clsf.get_pred([instance])[0]
        instance_for_onmt = self.onmt_model.clean_text(instance, only_upper=False)
        paraphrases = self.ps.generate_paraphrases(instance_for_onmt, topk=topk, edit_distance_cutoff=4, threshold=threshold)
        texts = self.tokenizer.clean_for_model(self.tokenizer.clean_for_humans([x[0] for x in paraphrases]))
        preds = clsf.get_pred(texts)
        fs = [(texts[i], paraphrases[i][1]) for i in np.where(preds != orig_pred)[0]]
        return fs

