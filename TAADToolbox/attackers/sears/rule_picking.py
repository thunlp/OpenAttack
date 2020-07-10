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
        # if flips.shape[0] == 0 or rule_precsupports[i] == 0 or flips.shape[0] < min_flips:
        #     disqualified.add(i)
        else:
            if rule_precsupports[i] == 0:
                precision = 0
            else:
                precision = flips.shape[0] / float(rule_precsupports[i])
            rule_precisions.append(precision)
            if precision < min_precision:
                disqualified.add(i)
    # for i, scores in enumerate(rule_scores):
    #     if (scores < threshold).mean() > 0.1:
    #         disqualified.add(i)

    while len(chosen_rules) != k:
        best_gain = (-10000000000, -100000000000)
        chose_something = False
        for i, (scores, flips, scores_on_all) in enumerate(zip(rule_scores, rule_flips, frequent_scores_on_all)):
            if i in chosen_rules or i in disqualified:
                continue
            if flips.shape[0] == 0:
                continue
                # scores = scores.copy()
                # scores[scores > -3] = 1
                # scores[(scores < -3) * (scores > -7)] = 0
                # scores[scores < -7] = -1
                # print scores.sum()
            # bad = (scores < threshold).sum()
            bad = (scores_on_all < min_bad_score).mean()
            good = np.where(scores >= min_bad_score)[0]
            scores = scores.copy()
            # scores[good] = 0
            # scores = (1 - bad) * np.exp(scores)
            scores = np.exp(scores) - LAMBDA
            # gain = 7999 - bad
            # gain = compute_gain(current_score[flips[good]], scores[good], metric=metric)
            gain = compute_gain(current_score[flips], scores, metric=metric)
            if gain == 0:
                continue
            subtract = (np.exp(scores_on_all) - LAMBDA)
            subtract = subtract[subtract < 0].sum()
            gain = gain + subtract
            # gain = gain + scores[scores < 0].sum()
            # gain = .005 * (7999 - bad) + compute_gain(current_score[flips[good]], scores[good], metric=metric)
            if gain > best_gain[1]:
                best_gain = (i, gain)
                chose_something = True
                # print(best_gain, subtract, scores[scores > 0].sum(), gain - subtract)
            # print
        if not chose_something:
            break
        if best_gain[1] <= 0:
            break
        # print('AYOO')
        chosen = best_gain[0]
        # print best_gain
        # print 'chosen', chosen
        bad = (frequent_scores_on_all[chosen] < min_bad_score).mean()
        chosen_rules.append(chosen)
        # scores = rule_scores[chosen].copy()
        scores = rule_scores[chosen].copy()
        scores = np.exp(scores) - LAMBDA
        current_score[rule_flips[chosen]] = compute_new(current_score[rule_flips[chosen]], scores, metric=metric)

        # good = np.where(scores >= min_bad_score)[0]
        # # scores[good] = 0
        # # scores = (1 - bad) * np.exp(scores[good])
        # scores = np.exp(scores[good]) - LAMBDA
        # current_score[rule_flips[chosen][good]] = compute_new(current_score[rule_flips[chosen][good]], scores, metric=metric)
        # print current_score.sum()


    return chosen_rules

    for i, (scores, flips) in enumerate(zip(rule_scores, rule_flips)):
        bad = (scores < threshold).sum()
        good = np.exp(scores[scores >= threshold]).sum()
        stats.append((bad, good))
        rscores.append(val_length - 10 * bad + good)

    chosen = list(np.argsort(rscores)[-k:])
    chosen.reverse()
    for c in chosen:
        print(stats[c], rscores[c])
    return chosen

def disqualify_rules(rule_scores, rule_flips, rule_precsupports, min_precision=0,
                     min_flips=0, min_bad_score=-99999, max_bad_proportion = 1,
                     max_bad_sum=99999999):
    disqualified = set()
    for i, (scores, flips) in enumerate(zip(rule_scores, rule_flips)):
        bad = (scores < min_bad_score)
        # if bad.mean() > max_bad_proportion:
        #     disqualified.add(i)
        #if np.percentile(scores, max_bad_proportion * 100) < min_bad_score:
        #    disqualified.add(i)
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
    # coverage * score
    current_score = np.zeros(val_length)
    chosen_rules = []
    rule_precisions = []
    to_use = rule_scores if frequent_scores_on_all is None else frequent_scores_on_all
    '''if disqualified is None:
        disqualified = disqualify_rules(to_use, rule_flips, rule_precsupports,
                                        min_precision, min_flips, min_bad_score, max_bad_proportion, max_bad_sum)'''
    # print(len(disqualified))
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
            '''if i in chosen_rules or i in disqualified:
                continue'''
            if i in chosen_rules:
                continue
            if flips.shape[0] == 0:
                continue
            if exp:
                scores = np.exp(scores)
                # scores = scores.copy()
                # scores[scores > -3] = 1
                # scores[(scores < -3) * (scores > -7)] = 0
                # scores[scores < -7] = -1
                # print scores.sum()
            gain = compute_gain(current_score[flips], scores, metric=metric)
            if gain > best_gain[1]:
                # print i, gain
                best_gain = (i, gain)
            # print
        if best_gain[1] <= 0:
            break
        chosen = best_gain[0]
        # print best_gain
        # print 'chosen', chosen
        chosen_rules.append(chosen)
        if exp:
            current_score[rule_flips[chosen]] = compute_new(current_score[rule_flips[chosen]], np.exp(rule_scores[chosen]), metric=metric)
        else:
            # scores = rule_scores[chosen].copy()
            # scores[scores > -3] = 1
            # scores[(scores < -3) * (scores > -7)] = 0
            # scores[scores < -7] = -1
            # print (scores == -3).sum(), (scores ==0).sum(), (scores == 1).sum(), scores.sum()
            scores = rule_scores[chosen]
            current_score[rule_flips[chosen]] = compute_new(current_score[rule_flips[chosen]], scores, metric=metric)
        # print current_score.sum()
#         current_score[rule_flips[chosen]] = np.maximum(current_score[rule_flips[chosen]], np.exp(rule_scores[chosen]))
    return chosen_rules

def choose_rules_2(rule_scores, rule_flips, rule_supports, rule_precsupports, val_length, k=10, min_flips=0, metric='max'):
    # Coverage,  score, precision
    current_score = np.zeros(val_length)
    rule_precisions = []
    for r in range(len(rule_flips)):
        if rule_flips[r].shape[0] == 0 or rule_flips[r].shape[0] < min_flips:
            rule_precisions.append(0)
            continue
        rule_precisions.append(rule_flips[r].shape[0] / rule_precsupports[r])
    rule_precisions = np.array(rule_precisions)
    chosen_rules = []
    while len(chosen_rules) != k:
        best_gain = (-1, -1)
        for i, (scores, flips, precision) in enumerate(zip(rule_scores, rule_flips, rule_precisions)):
            if i in chosen_rules:
                continue
            if flips.shape[0] == 0:
                continue
            scores = np.exp(scores) * precision
            gain = compute_gain(current_score[flips], scores, metric=metric)
            if gain > best_gain[1]:
                best_gain = (i, gain)
        if best_gain[1] <= 0:
            break
        chosen = best_gain[0]
#         print best_gain[1]
        chosen_rules.append(chosen)
        current_score[rule_flips[chosen]] = compute_new(current_score[rule_flips[chosen]], rule_precisions[chosen] * np.exp(rule_scores[chosen]), metric=metric)
    return chosen_rules
def choose_rules_3(rule_scores, rule_flips, rule_supports, rule_precsupports, min_flip, k=10):
    # Top precision * mean score given min_flips
    assert min_flip > 0
    rule_precisions = []
    scores = []
    for r in range(len(rule_flips)):
        if rule_flips[r].shape[0] < min_flip:
            scores.append(0)
            continue
        precision = rule_flips[r].shape[0] / rule_precsupports[r]
        score = precision * np.exp(rule_scores[r]).mean()
        scores.append(-score)
    return np.argsort(scores)[:k]
def choose_rules_4(rule_scores, rule_flips, rule_supports,rule_precsupports, val_length, min_flip, k=10, min_score=0, metric='max'):
    # Coverage,  precision (no score)
    assert min_flip > 0
    current_score = np.zeros(val_length)
    rule_precisions = []
    for r in range(len(rule_flips)):
        if rule_flips[r].shape[0] == 0:
            rule_precisions.append(0)
            continue
        if np.mean(np.exp(rule_scores[r])) < min_score or rule_flips[r].shape[0] < min_flip:
            rule_precisions.append(0)
            continue
        rule_precisions.append(rule_flips[r].shape[0] / rule_precsupports[r])
    rule_precisions = np.array(rule_precisions)
    chosen_rules = []
    while len(chosen_rules) != k:
        best_gain = (-1, -1)
        for i, (scores, flips, precision) in enumerate(zip(rule_scores, rule_flips, rule_precisions)):
            if i in chosen_rules:
                continue
            if flips.shape[0] == 0:
                continue
            scores = np.ones(len(scores)) * precision
            gain = compute_gain(current_score[flips], scores, metric=metric)
            if gain > best_gain[1]:
                best_gain = (i, gain)
        if best_gain[1] <= 0:
            break
        chosen = best_gain[0]
        chosen_rules.append(chosen)
        current_score[rule_flips[chosen]] = compute_new(current_score[rule_flips[chosen]], rule_precisions[chosen] * np.exp(rule_scores[chosen]), metric=metric)
    return chosen_rules
