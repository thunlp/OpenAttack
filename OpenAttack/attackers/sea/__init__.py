import os
import pickle
import numpy as np
import collections
from tqdm import tqdm
from ...utils import check_parameters
from ...text_processors import DefaultTextProcessor
from ...attacker import Attacker
from ...data_manager import DataManager

DEFAULT_TO_PATHS = ['english_french_model_acc_71.05_ppl_3.71_e13.pt', 'english_portuguese_model_acc_70.75_ppl_4.32_e13.pt']
DEFAULT_BACK_PATHS = ['french_english_model_acc_68.51_ppl_4.43_e13.pt', 'portuguese_english_model_acc_69.93_ppl_5.04_e13.pt']

DEFAULT_CONFIG = {
    "processor": DefaultTextProcessor(),
    "rules": None,
}

TRAIN_CONFIG = {
    "processor": DefaultTextProcessor(),
    "gpu_id": 0,
    "cuda": True,
    "topk": 200, 
    "threshold": -15.,
    "min_freq": 0.005,
    "ngram_size": 4,
}

class SEAAttacker(Attacker):
    def __init__(self, **kwargs):
        """
        :param tuple rules: A tuple of parameters related to rules used in SEAAttacker. The tuple is returned by classmethod get_rules(). **Default:** None.
        :param TextProcessor processor: Text processor used in this attacker. **Default:** :any:`DefaultTextProcessor`.

        :Package Requirements:
            * torch
            * torchtext
            * editdistance
        :Data Requirements: :py:data:`.AttackAssist.TranslationModels`
        :Classifier Capacity: Decision

        Semantically Equivalent Adversarial Rules for Debugging NLP Models. Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. ACL 2018.
        `[pdf] <https://aclweb.org/anthology/P18-1079>`__
        `[code] <https://github.com/marcotcr/sears>`__

        """
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
        self.tokenizer = self.replace_rules.Tokenizer(self.config["processor"])
        self.x = self.config["rules"][0]
        self.really_frequent_rules = self.config["rules"][1]
        self.frequent_rules = self.config["rules"][2]
        self.rule_flips = self.config["rules"][3]
        self.token_right = self.config["rules"][4]



    def __call__(self, clsf, x_orig, target=None):
        x_orig = x_orig.lower()
        if target is None:
            targeted = False
            target = clsf.get_pred([x_orig])[0]  
        else:
            targeted = True


        x_orig = [self.onmt_model.clean_text(x_orig.lower(), only_upper=False)]
        x_orig = self.tokenizer.tokenize(x_orig)[0]
        for r in self.x:
            rid = self.really_frequent_rules[r]
            rule = self.frequent_rules[rid]
            for f in self.rule_flips[rid][:2]:
                if self.token_right[f] != x_orig:
                    continue
                new = rule.apply(self.token_right[f])[1]
                ans = clsf.get_pred([new])[0]
                if targeted:
                    if ans == target:
                        return new, ans
                else:
                    if ans != target:
                        return new, ans
        
        return None

    @classmethod
    def get_rules(cls, clsf, sentence_list, **kwargs):
        """
        :param Classifier clsf: Your classifier.
        :param Dataset sentence_list: A list of sentences.
        :param TextProcessor processor: Text processor used in this attacker. **Default:** :any:`DefaultTextProcessor`.
        :param bool cuda: Use gpu or not. **Default:** :True.
        :param int gpu_id: gpu_id. **Default:** :0.
        :param int topk: The number of flips to find. **Default:** :200.
        :param float threshold: The threshold used in finding flips module. **Default:** :-15.0.
        :param float min_freq: Frequency threshold in finding rules module. **Default:** :0.005.
        :param int ngram_size: The length of replaced part in rules. **Default:** :4.

        :Package Requirements:
            * torch
            * torchtext
            * editdistance
        :Data Requirements: :py:data:`.AttackAssist.TranslationModels`

        """
        from . import paraphrase_scorer
        from . import onmt_model
        from . import replace_rules
        from . import rule_picking

        def find_flips(instance, clsf, ps, tokenizer, topk=10, threshold=-10):
            orig_pred = clsf.get_pred([instance])[0]
            instance_for_onmt = onmt_model.clean_text(instance, only_upper=False)
            paraphrases = ps.generate_paraphrases(instance_for_onmt, topk=topk, edit_distance_cutoff=5, threshold=threshold)
            texts = tokenizer.clean_for_model(tokenizer.clean_for_humans([x[0] for x in paraphrases]))
            preds = clsf.get_pred(texts)

            fs = [(texts[i], paraphrases[i][1]) for i in np.where(preds != orig_pred)[0]]
            return fs

        config = TRAIN_CONFIG.copy()
        config.update(kwargs)
        check_parameters(TRAIN_CONFIG.keys(), config)

        model_path = DataManager.load("AttackAssist.TranslationModels")

        to_paths = [model_path[x] for x in DEFAULT_TO_PATHS]
        back_paths = [model_path[x] for x in DEFAULT_BACK_PATHS]
        ps = paraphrase_scorer.ParaphraseScorer(to_paths=to_paths, back_paths=back_paths, gpu_id=config["gpu_id"], cuda=config["cuda"])
        tokenizer = replace_rules.Tokenizer(config["processor"])

        val_for_onmt = [onmt_model.clean_text(sentence.x.lower(), only_upper=False) for sentence in sentence_list]

        orig_scores = {}
        flips = collections.defaultdict(lambda: [])
        right_val = val_for_onmt
        right_preds = clsf.get_pred(right_val)

        for i, sentence in tqdm(enumerate(right_val)):
            if sentence in flips:
                continue
            fs = find_flips(sentence, clsf, ps, tokenizer, topk=config["topk"], threshold=config["threshold"])
            flips[sentence].extend([x[0] for x in fs])

        tr2 = replace_rules.TextToReplaceRules(config["processor"], right_val, [], 
                                                    min_freq=config["min_freq"], min_flip=0.00, ngram_size=config["ngram_size"])
        frequent_rules = []
        rule_idx = {}
        rule_flips = {}
        for z, f in enumerate(flips):
            rules = tr2.compute_rules(f, flips[f], use_pos=True, use_tags=False)
            for rs in rules:
                for r in rs:
                    if r.hash() not in rule_idx:
                        i = len(rule_idx)
                        rule_idx[r.hash()] = i
                        rule_flips[i] = []
                        frequent_rules.append(r)
                    i = rule_idx[r.hash()]
                    rule_flips[i].append(z)

        token_right = tokenizer.tokenize(right_val)
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
            orig_scores[i] = ps.score_sentences(t, [t])[0]
        ps_scores = {}
        ps.last = None
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
                        score = ps.score_sentences(o, [n])[0]
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

        disqualified = rule_picking.disqualify_rules(rule_scores, frequent_flips,
                          rule_precsupports, 
                      min_precision=0.0, min_flips=6, 
                         min_bad_score=threshold, max_bad_proportion=.10,
                          max_bad_sum=999999)

        x = rule_picking.choose_rules_coverage(rule_flip_scores, frequent_flips, None,
                          None, len(right_preds),
                                frequent_scores_on_all=None, k=10, metric='max',
                      min_precision=0.0, min_flips=0, exp=True,
                         min_bad_score=threshold, max_bad_proportion=.1,
                          max_bad_sum=999999,
                         disqualified=disqualified,
                         start_from=[])
        return (x, really_frequent_rules, frequent_rules, rule_flips, token_right)