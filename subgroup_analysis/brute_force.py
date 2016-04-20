# coding: utf-8
import math
import itertools
import statistics
from enum import Enum
import zipfile

import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower


# Configuration
P_VALUE_THRESHOLD = 0.05
TRT_EFFECT_THRESHOLD = -0.6
NUMBER_THRESHOLD = (60, 180)
POWER_THRESHOLD = 0.8


class Comparison(Enum):
    smaller = -1
    equal = 0
    larger = 1

def get_quartiles(ary):
    q1 = np.percentile(ary, 25)
    q2 = np.percentile(ary, 50)
    q3 = np.percentile(ary, 75)

    return q1, q2, q3

# Generate all possible rules
def possible_rules(rule_length, patients):
    discrete_values = [(0,), (1,), (2,), (0, 1), (1, 2)]
    for param in range(1, 21):
        for dv in discrete_values:
            yield (param, dv, Comparison.equal)

    for param in range(21, 41):
        continuous_values = get_quartiles(patients['x{}'.format(param)])
        for cv in continuous_values:
            yield (param, cv, Comparison.smaller)
            yield (param, cv, Comparison.larger)

    '''
    for c in itertools.combinations(parameters, rule_length):
        d_len = rule_length
        for dv in itertools.permutations(discrete_values, d_len):
            yield (c, dv)
    '''

# Grouping method
def check_group(patient, rule):
    idx, val, flag = rule

    achieve_num = 0
    if flag == Comparison.equal:
        if patient.loc['x{}'.format(idx)] in val:
            achieve_num += 1

    elif flag == Comparison.smaller:
        if patient.loc['x{}'.format(idx)] <= val:
            achieve_num += 1

    elif flag == Comparison.larger:
        if patient.loc['x{}'.format(idx)] > val:
            achieve_num += 1

    if achieve_num == 1:
        return 1

    return 0

def evaluation(d):

    # Subgroup
    gp = d[d['group'] == 1]
    gp_1 = gp[gp['trt'] == 1]
    gp_0 = gp[gp['trt'] == 0]

    gp_1_num = len(gp_1)
    gp_0_num = len(gp_0)
    gp_num = gp_1_num + gp_0_num
    if gp_1_num < 2 or gp_0_num < 2:
        return None

    gp_trt_effect = statistics.mean(gp_1['y']) - statistics.mean(gp_0['y'])

    gp_1_stdev = statistics.stdev(gp_1['y'])
    gp_0_stdev = statistics.stdev(gp_0['y'])
    tmp_1 = (gp_1_num-1)*math.pow(gp_1_stdev, 2)
    tmp_0 = (gp_0_num-1)*math.pow(gp_0_stdev, 2)
    gp_stdev = math.sqrt((tmp_1 + tmp_0)/(gp_1_num + gp_0_num - 2))

    gp_ratio = gp_0_num / gp_1_num
    _, gp_p = stats.ttest_ind(gp_1['y'], gp_0['y'], equal_var=False)
    gp_p /= 2

    gp_effect_size = gp_trt_effect/gp_stdev
    # power = TTestIndPower.power(-effect_size, new_trt_number, P_VALUE_THRESHOLD, ratio, new_trt_number+old_trt_number-2, alternative="larger")

    # Nongroup
    gn = d[d['group'] == 0]
    gn_1 = gn[gn['trt'] == 1]
    gn_0 = gn[gn['trt'] == 0]

    gn_1_num = len(gn_1)
    gn_0_num = len(gn_0)
    gn_num = gn_1_num + gn_0_num
    if gn_1_num < 2 or gn_0_num < 2:
        return None

    gn_trt_effect = statistics.mean(gn_1['y']) - statistics.mean(gn_0['y'])

    gn_1_stdev = statistics.stdev(gn_1['y'])
    gn_0_stdev = statistics.stdev(gn_0['y'])
    tmp_1 = (gn_1_num-1)*math.pow(gn_1_stdev, 2)
    tmp_0 = (gn_0_num-1)*math.pow(gn_0_stdev, 2)
    gn_stdev = math.sqrt((tmp_1 + tmp_0)/(gn_1_num + gn_0_num - 2))

    gn_effect_size = gn_trt_effect/gn_stdev

    _, gn_p = stats.ttest_ind(gn_1['y'], gn_0['y'], equal_var=False)
    gn_p /= 2

    # Inter
    trt_effect_1 = statistics.mean(gp_1['y']) - statistics.mean(gn_1['y'])
    trt_effect_0 = statistics.mean(gp_0['y']) - statistics.mean(gn_0['y'])

    _, p_1 = stats.ttest_ind(gp_1['y'], gn_1['y'], equal_var=False)
    _, p_0 = stats.ttest_ind(gp_0['y'], gn_0['y'], equal_var=False)

    # Filter
    if gp_trt_effect > TRT_EFFECT_THRESHOLD:
        return None
    elif trt_effect_1 > TRT_EFFECT_THRESHOLD:
        return None
    elif gp_p > P_VALUE_THRESHOLD:
        return None
    elif p_1 > P_VALUE_THRESHOLD:
        return None
    elif gp_num < NUMBER_THRESHOLD[0] or gp_num > NUMBER_THRESHOLD[1]:
        return None

    gp_1_mean = statistics.mean(gp_1['y'])
    gp_0_mean = statistics.mean(gp_0['y'])
    gn_1_mean = statistics.mean(gn_1['y'])
    gn_0_mean = statistics.mean(gn_0['y'])
    return gp_effect_size, gp_num, gp_trt_effect, gp_p, gp_1_mean, gp_1_num, gp_0_mean, gn_trt_effect, gn_p, gn_1_mean, gn_1_num, gn_0_mean,trt_effect_1, p_1, trt_effect_0, p_0

def h0_test(d):

    trt_1 = d[d['trt'] == 1]
    trt_0 = d[d['trt'] == 0]

    trt_effect = statistics.mean(trt_1['y']) - statistics.mean(trt_0['y'])
    z_stat, p_val = stats.ranksums(trt_1['y'], trt_0['y'])

    #print('H0 treatment effect: {}, p-value: {}'.format(trt_effect, p_val))
    return trt_effect, p_val

def find_good_rules(t):
    # Load parameters
    did, d, rule_len = t

    gp_effect_size_rank = []
    gp_trt_effect_rank = []
    trt_effect_1_rank = []

    h0_trt_effect, h0_p_val = h0_test(d)

    # Iteratively get one of the possible rules
    for rule in possible_rules(rule_len, d):

        for i in range(1, 241):
            patient = d[d['id'] == i].iloc[0]
            d.loc[d['id'] == i, 'group'] = check_group(patient, rule)

        # Evaluate the treatment effect according each rule
        parts = evaluation(d)
        if parts is None:
            continue

        # Caculate the socre
        gp_effect_size, gp_num, gp_trt_effect, gp_p, gp_1_mean, gp_1_num, gp_0_mean, gn_trt_effect, gn_p, gn_1_mean, gn_1_num, gn_0_mean,trt_effect_1, p_1, trt_effect_0, p_0 = parts

        # Check all conditions
        # h0_p_val > P_VALUE_THRESHOLD:

        gp_effect_size_rank.append(((rule, parts), gp_effect_size))
        gp_trt_effect_rank.append(((rule, parts), gp_trt_effect))
        trt_effect_1_rank.append(((rule, parts), trt_effect_1))

    # Ranking
    gp_effect_size_rank = sorted(gp_effect_size_rank, key=lambda x: x[1])
    gp_trt_effect_rank = sorted(gp_trt_effect_rank, key=lambda x: x[1])
    trt_effect_1_rank = sorted(trt_effect_1_rank, key=lambda x: x[1])

    rank_dict = dict()
    for i in range(0, 10):

        try:
            item = gp_effect_size_rank[i]
            if item[0] in rank_dict:
                rank_dict[item[0]] += 10-i
            else:
                rank_dict[item[0]] = 10-i
        except:
            pass

        try:
            item = gp_trt_effect_rank[i]
            if item[0] in rank_dict:
                rank_dict[item[0]] += 10-i
            else:
                rank_dict[item[0]] = 10-i
        except:
            pass

        try:
            item = trt_effect_1_rank[i]
            if item[0] in rank_dict:
                rank_dict[item[0]] += 10-i
            else:
                rank_dict[item[0]] = 10-i
        except:
            pass

    rank = []
    for k, v in rank_dict.items():
        rank.append((k, v))
    rank = sorted(rank, reverse=True, key=lambda x: x[1])

    # Record the best rule for this dataset
    if len(rank) >= 1:
        best_rule = rank[0][0][0]
        best_score = rank[0][1]

        print('Dataset {}: {} ({})'.format(did, best_rule, best_score))

        # Preprocess rank
        preprocess = []
        for r in rank:
            gp_effect_size, gp_num, gp_trt_effect, gp_p, gp_1_mean, gp_1_num, gp_0_mean, gn_trt_effect, gn_p, gn_1_mean, gn_1_num, gn_0_mean,trt_effect_1, p_1, trt_effect_0, p_0 = r[0][1]

            preprocess.append((r[0][0], gp_effect_size, gp_num, gp_trt_effect, gp_p, gp_1_mean, gp_1_num, gp_0_mean, gn_trt_effect, gn_p, gn_1_mean, gn_1_num, gn_0_mean,trt_effect_1, p_1, trt_effect_0, p_0, r[1]))

        # Save to CSV
        import pandas
        df = pandas.DataFrame(preprocess, columns=['rule', 'gp_effect_size', 'gp_num', 'gp_trt_effect', 'gp_p', 'gp_1_mean', 'gp_1_num', 'gp_0_mean', 'gn_trt_effect', 'gn_p', 'gn_1_mean', 'gn_1_num', 'gn_0_mean', 'trt_effect_1', 'p_1', 'trt_effect_0', 'p_0', 'score'])
        df.to_csv('rules/{}.csv'.format(did))

        return df
