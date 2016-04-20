# coding: utf-8
import time
import math
import itertools
import statistics
import zipfile

import pandas
import numpy as np
from scipy import stats

# Configuration
P_VALUE_THRESHOLD = 0.001
TRT_EFFECT_THRESHOLD = -0.6
NUMBER_THRESHOLD = (60, 180)

# Generate all possible rules
def possible_rules(rule_length):
    # Only use x1-x20
    parameters = range(1, 21)
    discrete_values = [(0,), (1,), (2,), (0, 1), (1, 2)]

    for c in itertools.combinations(parameters, rule_length):
        d_len = rule_length

        for dv in itertools.permutations(discrete_values, d_len):
            yield (c, dv)

# Grouping method
def check_group(patient, rule):
    indexes, values = rule

    achieve_num = 0
    for i in range(len(indexes)):
        idx = indexes[i]
        val = values[i]

        # Discrete rule
        if type(val) == tuple:
            if patient.loc['x{}'.format(idx)] in val:
                achieve_num += 1

        # Continuous rule
        else:
            print('Continuous rule.')

    if achieve_num == len(indexes):
        return 1

    return 0

def evaluation(d):

    # Subgroup
    subgroup = d[d['group'] == 1]
    new_trt_group = subgroup[subgroup['trt'] == 1]
    old_trt_group = subgroup[subgroup['trt'] == 0]

    if len(new_trt_group) == 0 or len(old_trt_group) == 0:
        return None

    trt_effect = statistics.mean(new_trt_group['y']) - statistics.mean(old_trt_group['y'])
    number = len(new_trt_group) + len(old_trt_group)
    z_stat, p_val = stats.ranksums(new_trt_group['y'], old_trt_group['y'])

    # Nongroup
    nongroup = d[d['group'] == 0]
    new_trt_group = nongroup[nongroup['trt'] == 1]
    old_trt_group = nongroup[nongroup['trt'] == 0]

    if len(new_trt_group) == 0 or len(old_trt_group) == 0:
        return None

    nongroup_mean = statistics.mean(new_trt_group['y']) - statistics.mean(old_trt_group['y'])

    # Filter
    if trt_effect > TRT_EFFECT_THRESHOLD:
        return None
    elif p_val > P_VALUE_THRESHOLD:
        return None
    elif number < NUMBER_THRESHOLD[0] or number > NUMBER_THRESHOLD[1]:
        return None

    ratio = (trt_effect/nongroup_mean) if nongroup_mean != 0 else 0
    minus = trt_effect - nongroup_mean

    return ratio, minus, trt_effect, number, p_val

def h0_check(d):

    trt_1 = d[d['trt'] == 1]
    trt_0 = d[d['trt'] == 0]

    trt_effect = statistics.mean(trt_1['y']) - statistics.mean(trt_0['y'])
    z_stat, p_val = stats.ranksums(trt_1['y'], trt_0['y'])

    if trt_effect > TRT_EFFECT_THRESHOLD:
        return None
    elif p_val > P_VALUE_THRESHOLD:
        return None

    return trt_effect

def update_best_score(score, best_score):
    if score == 0:
        return False

    if score < 0 and score < best_score:
        return True

    if score > 0 and best_score >= 0 and score > best_score:
        return True


def brute_force(t):
    # Load parameters
    df, start_dataset, end_dataset, rule_len = t

    # Set default group number to every patients
    if 'group' not in df:
        df['group'] = 0

    # Record the best rule for each dataset
    best_rules = []

    for did in range(start_dataset, end_dataset+1):
        d = df[df['dataset'] == did].copy()

        ratio_rank = []
        minus_rank = []
        treat_rank = []
        number_rank = []
        p_rank = []

        h0 = h0_check(d)
        if h0 is not None:
            print(h0)

        # Iteratively get one of the possible rules
        for rule in possible_rules(rule_len):

            for i in range(1, 241):
                patient = d[d['id'] == i].iloc[0]
                d.loc[d['id'] == i, 'group'] = check_group(patient, rule)

            # Evaluate the treatment effect according each rule
            parts = evaluation(d)
            if parts is None:
                continue

            # Caculate the socre
            ratio, minus, trt_effect, number, p_val = parts

            ratio_rank.append((rule, ratio, math.fabs(ratio)))
            minus_rank.append((rule, minus, minus))
            treat_rank.append((rule, trt_effect, trt_effect))
            number_rank.append((rule, number, math.fabs(100-number)))
            p_rank.append((rule, p_val, p_val))

            # Compare with the absolute value
            '''
            if update_best_score(score, best_score):
                best_score = score
                best_rule = rule

                print(['x{}'.format(tmp-3) for tmp in rule[0]], rule[1], ratio, trt_effect, number, p_val)
            '''

        # Ranking
        # Best ratio
        ratio_rank = sorted(ratio_rank, reverse=True, key = lambda x : x[2])
        minus_rank = sorted(minus_rank, reverse=False, key = lambda x : x[2])
        treat_rank = sorted(treat_rank, reverse=False, key = lambda x : x[2])
        number_rank = sorted(number_rank, reverse=False, key = lambda x : x[2])
        p_rank = sorted(p_rank, reverse=False, key = lambda x : x[2])

        print('Top 10 Ratio: 絕對值越高越好')
        for r in ratio_rank[:10]:
            param = r[0][0][0]
            val = r[0][1][0]
            print('x{} = {}: {}'.format(param, val, r[1]))

        print('\nTop 10 Minus: 越低越好')
        for r in minus_rank[:10]:
            param = r[0][0][0]
            val = r[0][1][0]
            print('x{} = {}: {}'.format(param, val, r[1]))

        print('\nTop 10 Treatment Effect: 越低越好')
        for r in treat_rank[:10]:
            param = r[0][0][0]
            val = r[0][1][0]
            print('x{} = {}: {}'.format(param, val, r[1]))

        print('\nTop 10 Number: 越接近 100 越好')
        for r in number_rank[:10]:
            param = r[0][0][0]
            val = r[0][1][0]
            print('x{} = {}: {}'.format(param, val, r[1]))

        print('\nTop 10 P-Value: 越低越好')
        for r in p_rank[:10]:
            param = r[0][0][0]
            val = r[0][1][0]
            print('x{} = {}: {}'.format(param, val, r[1]))

        # Record the best rule for this dataset
        '''
        if best_score != 0:
            best_rules.append((did, best_score, best_rule))
        else:
            best_rules.append((did, 0, None))

        # Dvide the patients into groups according to the best rule
        for i in range(1, 241):
            patient = df[(df['dataset'] == did) & (df['id'] == i)].iloc[0]

            if best_rule is None:
                g = 0
            else:
                g = check_group(patient, best_rule)

            df.loc[(df['dataset'] == did) & (df['id'] == i), 'group'] = g
        '''

    return df, best_rules
