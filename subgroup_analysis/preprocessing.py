# coding: utf-8
import math
import itertools
import statistics
from enum import Enum
import zipfile

import numpy as np
from scipy import stats
from scipy.stats import norm
from statsmodels.stats.power import TTestIndPower


# Configuration
P_VALUE_THRESHOLD = 0.05
TRT_EFFECT_THRESHOLD = -0.6
NUMBER_THRESHOLD = (60, 180)
POWER_THRESHOLD = 0.8


def get_quartiles(ary):
    unit = 100/12
    return [i*unit for i in range(1, 12)]

# Generate all possible rules
def get_rule_1(patients):
    discrete_values = [(0,), (1,), (2,), (0, 1), (1, 2)]
    for param in range(1, 21):
        for dv in discrete_values:
            yield (param, dv, 'equal')

    for param in range(21, 41):
        sorted_df = patients[patients['trt'] == 1].sort_values('x{}'.format(param))
        slope, _, _, _, _ = stats.linregress(sorted_df['x{}'.format(param)], sorted_df['y'])

        continuous_values = get_quartiles(patients['x{}'.format(param)])
        for cv in continuous_values:
            if slope < 0:
                yield (param, cv, 'larger')
            elif slope > 0:
                yield (param, cv, 'smaller')

    yield ((4, (1, 2), 'equal'), (22, 57.7, 'larger'), 'and')

def get_rule_2(patients):
    rules = [r for r in get_rule_1(patients)]
    for i in range(0, len(rules)):
        for j in range(i, len(rules)):
            if rules[i][0] == rules[j][0]:
                continue

            yield (rules[i], rules[j], 'and')
            yield (rules[i], rules[j], 'or')

def get_cartesian_product(did):
    df = pandas.read_pickle('rules/{}_1.pickle'.format(did))
    single_rules = df['rule']

    for i in range(0, len(single_rules)):
        k = i+1
        while k < len(single_rules):
            yield (single_rules[i], single_rules[k], 'and')
            yield (single_rules[i], single_rules[k], 'or')
            k = k+1

def check_group_2(patient, rule):
    if (rule[2] == 'or') | (rule[2] == 'and'):
        rule_1, rule_2, flag = rule
        if flag == 'and':
            return (check_group_1(patient, rule_1) & check_group_1(patient, rule_2))

        elif flag == 'or':
             return (check_group_1(patient, rule_1) | check_group_1(patient, rule_2))
    else:
        return check_group_1(patient, rule)

def check_group_1(patient, rule):
    idx, val, flag = rule

    if flag == 'equal':
        if len(val) > 1:
            return ((patient['x{}'.format(idx)] == val[0])|(patient['x{}'.format(idx)] == val[1]))
        else:
            return patient['x{}'.format(idx)] == val

    elif flag == 'smaller':
        return patient['x{}'.format(idx)] <= val

    elif flag == 'larger':
         return patient['x{}'.format(idx)] > val

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
    _, gp_p = stats.ranksums(gp_1['y'], gp_0['y'])

    gp_effect_size = gp_trt_effect/gp_stdev

    tmp_1 = math.pow(gp_stdev, 2) / math.pow(gp_num, 0.5)
    gp_power = norm.cdf((-gp_trt_effect/tmp_1)-1.96)

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

    _, gn_p = stats.ranksums(gn_1['y'], gn_0['y'])

    # Inter
    trt_effect_1 = statistics.mean(gp_1['y']) - statistics.mean(gn_1['y'])
    trt_effect_0 = statistics.mean(gp_0['y']) - statistics.mean(gn_0['y'])

    _, p_1 = stats.ranksums(gp_1['y'], gn_1['y'])
    _, p_0 = stats.ranksums(gp_0['y'], gn_0['y'])


    tmp_1 = (gp_1_num-1)*math.pow(gp_1_stdev, 2)
    tmp_0 = (gn_1_num-1)*math.pow(gn_1_stdev, 2)
    stdev_1 = math.sqrt((tmp_1 + tmp_0)/(gp_1_num + gn_1_num - 2))

    tmp_1 = math.pow(stdev_1, 2) / math.pow(gp_1_num+gn_1_num, 0.5)
    power_1 = norm.cdf((-trt_effect_1/tmp_1)-1.96)


    # Filter
    if gp_trt_effect > TRT_EFFECT_THRESHOLD:
        return None
    #elif trt_effect_1 > TRT_EFFECT_THRESHOLD:
    #    return None
    elif gp_p > P_VALUE_THRESHOLD:
        return None
    #elif p_1 > P_VALUE_THRESHOLD:
    #    return None
    elif gp_power < POWER_THRESHOLD:
        return None
    elif power_1 < POWER_THRESHOLD:
        return None
    #elif gp_num < NUMBER_THRESHOLD[0] or gp_num > NUMBER_THRESHOLD[1]:
    #    return None

    gp_1_mean = statistics.mean(gp_1['y'])
    gp_0_mean = statistics.mean(gp_0['y'])
    gn_1_mean = statistics.mean(gn_1['y'])
    gn_0_mean = statistics.mean(gn_0['y'])
    return gp_effect_size, gp_num, gp_trt_effect, gp_p, gp_1_mean, gp_1_num, gp_0_mean, gn_trt_effect, gn_p, gn_1_mean, gn_1_num, gn_0_mean,trt_effect_1, p_1, trt_effect_0, p_0, gp_power, power_1

def find_good_rules(t):
    # Load parameters
    did, d, rule_len = t

    rules = []

    # Get all possible rules
    if rule_len == 1:
        possible_rules = [r for r in get_rule_1(d)]
    elif rule_len == 1.5:
        possible_rules = [r for r in get_cartesian_product(did)]
    elif rule_len == 2:
        possible_rules = [r for r in get_rule_2(d)]
    else:
        print('wrong parameter.')
        return

    for rule in possible_rules:
        d['group'] = 0
        labels = check_group_2(d, rule)
        d.loc[labels, 'group'] = 1

        # Evaluate the treatment effect according each rule
        parts = evaluation(d)
        if parts is None:
            continue

        gp_effect_size, gp_num, gp_trt_effect, gp_p, gp_1_mean, gp_1_num, gp_0_mean, gn_trt_effect, gn_p, gn_1_mean, gn_1_num, gn_0_mean,trt_effect_1, p_1, trt_effect_0, p_0, gp_power, power_1 = parts
        rules.append((rule, gp_effect_size, gp_num, gp_trt_effect, gp_p, gp_1_mean, gp_1_num, gp_0_mean, gn_trt_effect, gn_p, gn_1_mean, gn_1_num, gn_0_mean,trt_effect_1, p_1, trt_effect_0, p_0, gp_power, power_1))

    # Save to piackle
    import pandas
    rules = pandas.DataFrame(rules, columns=['rule', 'gp_effect_size', 'gp_num', 'gp_trt_effect', 'gp_p', 'gp_1_mean', 'gp_1_num', 'gp_0_mean', 'gn_trt_effect', 'gn_p', 'gn_1_mean', 'gn_1_num', 'gn_0_mean', 'trt_effect_1', 'p_1', 'trt_effect_0', 'p_0', 'gp_power', 'power_1'])
    rules.to_pickle('rules/{}_{}.pickle'.format(did, rule_len))
    rules.to_csv('rules/{}_{}.csv'.format(did, rule_len))
    return rules
