# coding: utf-8
import time
import math
import pandas
import numpy as np

# Configuration
TREATMENT_EFFECT_THRESHOLD = -0.8

# Rule generation functions
def repeated_selections(items, n):
    if n == 0: yield []
    else:
        for i in range(len(items)):
            for r in repeated_selections(items, n-1):
                yield [items[i]] + r

def combinations(items, n):
    if n == 0: yield []
    else:
        for i in range(len(items)):
            for c in combinations(items[i+1:], n-1):
                yield [items[i]] + c

def possible_rules(max_rule_len):
    items = range(4, 44)
    dis_values = [(0), (1), (2), (0, 1), (1, 2)] # Possible discrete values
    con_values = [(25), (-25), (50), (-50), (75), (-75)] # Possible continuous values

    for rule_len in range(1, max_rule_len+1):
        for c in combinations(items, rule_len):
            con_param_num = 0 # Count the number of continuous parameters
            for e in c:
                if e > 23:
                    con_param_num += 1
            dis_param_num = rule_len-con_param_num

            # Generate all possible rules
            for dv in repeated_selections(dis_values, dis_param_num):
                for cv in repeated_selections(con_values, con_param_num):
                    yield (c, dv+cv)

# Grouping method
def check_group(patient, rule):
    indexes, values = rule

    for i in range(len(indexes)):
        idx = indexes[i]
        val = values[i]

        # Discrete rule
        if type(val) == tuple:
            if patient[idx] in val:
                return 1

        # Discrete rule
        elif math.fabs(val) < 3:
            if patient[idx] == val:
                return 1

        # Continuous rule
        else:
            if val > 0:
                if patient[idx] > val:
                    return 1
            else:
                if patient[idx] < -val:
                    return 1

    return 0

def evaluation():
    y0 = 0.0
    n0 = 0
    y1 = 0.0
    n1 = 0

def brute_force(t):
    # Load parameters
    df, start_dataset, end_dataset, rule_len = t

    # Set default group number to every patients
    if 'group' not in df:
        df['group'] = 0

    # Record the best rule for each dataset
    best_rules = []

    for did in range(start_dataset, end_dataset+1):

        # Iteratively get one of the possible rules
        best_treatment_effect = 0
        best_rule = None
        for rule in possible_rules(rule_len):
            for i in range(1, 241):
                patient = df[(df['dataset'] == did) & (df['id'] == i)].iloc[0]

                g = check_group(patient, rule)
                df.loc[(df['dataset'] == did) & (df['id'] == i), 'group'] = g

                evaluation(df)

                    if patient['trt'] == 0:
                        y0 += patient['y']
                        n0 += 1
                    else:
                        y1 += patient['y']
                        n1 += 1

            # Evaluate the treatment effect according each rule

            y1 = y1/n1 if n1 != 0 else 0
            y0 = y0/n0 if n0 != 0 else 0
            treatment_effect = y1 - y0
            if treatment_effect < TREATMENT_EFFECT_THRESHOLD:
                if treatment_effect < best_treatment_effect:
                    best_treatment_effect = treatment_effect
                    best_rule = rule

        # Record the best rule for this dataset
        if best_treatment_effect != 0:
            best_rules.append((did, best_treatment_effect, best_rule))
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

    return df, best_rules
