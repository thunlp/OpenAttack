import numpy as np

def compute_gain(old_scores, new_scores, metric='max'):
    if metric == 'max':
        potential_scores = np.maximum(old_scores, new_scores)
    elif metric == 'sqrtsum':
        potential_scores = np.sqrt(old_scores + new_scores)
        old_scores = np.sqrt(old_scores)
    elif metric == 'logsum':
        potential_scores = np.log(old_scores + new_scores + 1)
        old_scores = np.log(old_scores + 1)
    else:
        raise 'fail, metric must be max, sqrtsum or logsum, was %s' % metric
    gain = potential_scores.sum() - old_scores.sum()
    return gain

def compute_new(old_scores, new_scores, metric='max'):
    if metric == 'max':
        new = np.maximum(old_scores, new_scores)
    elif metric == 'sqrtsum' or metric == 'logsum':
        new = old_scores + new_scores
    else:
        raise 'fail, metric must be max, sqrtsum or logsum, was %s' % metric
    return new

def choose_rules_with_penalties(rule_scores, rule_flips, rule_supports, rule_precsupports, val_length, frequent_scores_on_all, k=10,
                          metric='max', min_precision=0, min_flips=0, min_bad_score=-99999, max_bad_sum=999999,
                          max_bad_proportion=1, exp=True, LAMBDA=0.5):
    rscores = []
    stats = []
    current_score = np.zeros(val_length)
    chosen_rules = []
    rule_precisions = []
    disqualified = set()
    to_use = rule_scores if frequent_scores_on_all is None else frequent_scores_on_all
    for i, (scores, flips) in enumerate(zip(to_use, rule_flips)):
        bad = (scores < min_bad_score)
        if (np.exp(scores) - LAMBDA).sum() < 0:
            disqualified.add(i)
        if bad.mean() > max_bad_proportion:
            disqualified.add(i)
        if bad.sum() > max_bad_sum:
            disqualified.add(i)

        else:
            if rule_precsupports[i] == 0:
                precision = 0
            else:
                precision = flips.shape[0] / float(rule_precsupports[i])
            rule_precisions.append(precision)
            if precision < min_precision:
                disqualified.add(i)


    while len(chosen_rules) != k:
        best_gain = (-10000000000, -100000000000)
        chose_something = False
        for i, (scores, flips, scores_on_all) in enumerate(zip(rule_scores, rule_flips, frequent_scores_on_all)):
            if i in chosen_rules or i in disqualified:
                continue
            if flips.shape[0] == 0:
                continue

            bad = (scores_on_all < min_bad_score).mean()
            good = np.where(scores >= min_bad_score)[0]
            scores = scores.copy()

            scores = np.exp(scores) - LAMBDA

            gain = compute_gain(current_score[flips], scores, metric=metric)
            if gain == 0:
                continue
            subtract = (np.exp(scores_on_all) - LAMBDA)
            subtract = subtract[subtract < 0].sum()
            gain = gain + subtract

            if gain > best_gain[1]:
                best_gain = (i, gain)
                chose_something = True

        if not chose_something:
            break
        if best_gain[1] <= 0:
            break

        chosen = best_gain[0]

        bad = (frequent_scores_on_all[chosen] < min_bad_score).mean()
        chosen_rules.append(chosen)

        scores = rule_scores[chosen].copy()
        scores = np.exp(scores) - LAMBDA
        current_score[rule_flips[chosen]] = compute_new(current_score[rule_flips[chosen]], scores, metric=metric)




    return chosen_rules


def disqualify_rules(rule_scores, rule_flips, rule_precsupports, min_precision=0,
                     min_flips=0, min_bad_score=-99999, max_bad_proportion = 1,
                     max_bad_sum=99999999):
    disqualified = set()
    for i, (scores, flips) in enumerate(zip(rule_scores, rule_flips)):
        bad = (scores < min_bad_score)
        if bad.sum() > max_bad_sum:
            disqualified.add(i)
        if flips.shape[0] == 0 or rule_precsupports[i] == 0 or flips.shape[0] < min_flips:
            disqualified.add(i)
        else:
            if rule_precsupports[i] == 0:
                precision = 0
            else:
                precision = flips.shape[0] / float(rule_precsupports[i])
            if precision < min_precision:
                disqualified.add(i)
    return disqualified

def choose_rules_coverage(rule_scores, rule_flips, rule_supports, rule_precsupports, val_length, k=10,
                          metric='max', min_precision=0, min_flips=0, min_bad_score=-99999, max_bad_proportion = 1, max_bad_sum=99999999, exp=True,
                          frequent_scores_on_all=None, disqualified=None, start_from=None):
    current_score = np.zeros(val_length)
    chosen_rules = []
    rule_precisions = []
    to_use = rule_scores if frequent_scores_on_all is None else frequent_scores_on_all

    if start_from is not None:
        for chosen in start_from:
            chosen_rules.append(chosen)
            if exp:
                current_score[rule_flips[chosen]] = compute_new(current_score[rule_flips[chosen]], np.exp(rule_scores[chosen]), metric=metric)
            else:
                scores = rule_scores[chosen]
                current_score[rule_flips[chosen]] = compute_new(current_score[rule_flips[chosen]], scores, metric=metric)
    while len(chosen_rules) < k:
        best_gain = (-1, -1)
        for i, (scores, flips) in enumerate(zip(rule_scores, rule_flips)):

            if i in chosen_rules or i in disqualified:
                continue
            if flips.shape[0] == 0:
                continue
            if exp:
                scores = np.exp(scores)

            gain = compute_gain(current_score[flips], scores, metric=metric)
            if gain > best_gain[1]:
                best_gain = (i, gain)
        if best_gain[1] <= 0:
            break
        chosen = best_gain[0]

        chosen_rules.append(chosen)
        if exp:
            current_score[rule_flips[chosen]] = compute_new(current_score[rule_flips[chosen]], np.exp(rule_scores[chosen]), metric=metric)
        else:
            scores = rule_scores[chosen]
            current_score[rule_flips[chosen]] = compute_new(current_score[rule_flips[chosen]], scores, metric=metric)

    return chosen_rules

