
# coding: utf-8


# In[1]:

import time
import pandas
import statistics
import threading
import _thread
import numpy as np
import zipfile
from scipy import stats  



#data load
testing_data = pandas.read_csv("dataset/Data.csv")
training_data = pandas.read_csv("dataset/Training_Data.csv")
training_truth = pandas.read_csv("dataset/Training_Data_truth.csv")


# In[21]:

#make upload file
class brute_force:
    P_VALUE_THRESHOLD = 0.05
    TRT_EFFECT_THRESHOLD = -0.6
    def make_upload_file(start_dataset, end_dataset, all_rules):
        output_file = 'result_{}.csv'.format(name)
        ans = pandas.DataFrame([], columns=(['id'] + ['dataset_{}'.format(i) for i in range(1, 1201)]))
        ans['id'] = range(1, 241)
        name = 'ranksum'

        for did in range(start_dataset, end_dataset+1):
            
            #initial
            ans['dataset_{}'.format(did)] = 0
            d = df[df['dataset'] == did].copy()

                
            for i in range(1, 241):
                patient = d[d['id'] == i].iloc[0] 
                g = check_group(patient, rule)
                d.loc[d['id'] == i, 'group'] = g
                subgroup = d[d['group'] == 1]

                for i in subgroup['id']:
                    ans.loc[i-1, 'dataset_{}'.format(did)] = 1
                
        ans.to_csv(output_file)   
        zf = zipfile.ZipFile('result_{}.zip'.format(name), 'w', zipfile.ZIP_DEFLATED)
        zf.write('result_{}.csv'.format(name))



    # In[97]:

    # Configuration
    

    def evaluation(d):  
        # Subgroup
        subgroup = d[d['group'] == 1]
        new_trt_group = subgroup[subgroup['trt'] == 1]
        old_trt_group = subgroup[subgroup['trt'] == 0]
        
        if len(new_trt_group) == 0 or len(old_trt_group) == 0:
            return 0
        
        trt_effect = statistics.mean(new_trt_group['y']) - statistics.mean(old_trt_group['y'])
        number = len(new_trt_group) + len(old_trt_group)
        z_stat, p_val = stats.ranksums(new_trt_group['y'], old_trt_group['y'])
        
        # Nongroup
        nongroup = d[d['group'] == 0]
        new_trt_group = nongroup[nongroup['trt'] == 1]
        old_trt_group = nongroup[nongroup['trt'] == 0]
        
        if len(new_trt_group) == 0 or len(old_trt_group) == 0:
            return 0
        
        nongroup_trt_effect = statistics.mean(new_trt_group['y']) - statistics.mean(old_trt_group['y'])
        minus = (trt_effect-nongroup_trt_effect) if nongroup_trt_effect != 0 else 0
        ratio = (trt_effect/nongroup_trt_effect) if nongroup_trt_effect != 0 else 0
        z_stat, non_p_val = stats.ranksums(new_trt_group['y'], old_trt_group['y'])
       
        # Filter
        if trt_effect > TRT_EFFECT_THRESHOLD: 
            return 0
        elif p_val > P_VALUE_THRESHOLD:
            return 0
        
        # Number of people restriction
        if number < 50 or number > 150:
            return 0
        
        return ratio, minus, trt_effect, nongroup_trt_effect, number, p_val, non_p_val

    def update_topN(score, rule, topN_score, case):
        if score == 0:    #why??
            return topN_score
        
        if case == 1:     # 1 = abs higher would be better
            s = abs(score)
        elif case == 2:   # 2 = close to 100 would be better
            s = abs(score - 100)
        else:             
            s = score
        topN_score.sort()  
        
        if len(topN_score)>=10 :
            
            key = []
            if (case == 1 | case == 4):    #higher would be better
                if s > topN_score[0][0]:    
                    key = topN_score[0]
               
            else:       
                topN_score.reverse()    #lower score would be better
                if s < topN_score[0][0]:
                    key = topN_score[0]
            
            if key in topN_score:
                topN_score.remove(key)
            else:
                return topN_score
            
        topN_score.append([s, score, rule])
        
        return topN_score
        

    # Grouping method
    def check_group(patient, rule):
        index, values, tag = rule
        
        if tag == 0:
            for v in values:
                if patient[index] == v:
                    return 1
            return 0

        elif tag == 1:
            if patient[index] > values[0]:
                return 1
        else:
            if patient[index] < values[0]:
                return 1
        return 0
       

    def quartile(d):
        Q1 = np.percentile(d, 25)
        Q2 = np.percentile(d, 50)
        Q3 = np.percentile(d, 75)
        return [(Q1), (Q2), (Q3)]


    def possible_rules(d):
        con_items = range(4, 24)
        dis_items = range(24, 44)
        dis_value =[[0], [1], [2], [0, 1], [0, 2], [1, 2]]
        rules = []

        for item in con_items:
            for value in dis_value:
                rules.append([(item), value, 0])
                
        for item in dis_items:
            con_values = quartile(d['x'+str(item-3)])
            for value in con_values:
                rules.append([(item), [value], 1])
                rules.append([(item), [value], 2])
        rules.append([(32), [(51.3)], 2])

        return rules  

                
    def return_key(item):
        if item[2] == 0:
            symbol = '='
            key = 'x{} {} {}'.format((item[0]-3), symbol, item[1])
                
        else:
            if item[2] == 1:
                symbol = '>'
            elif item[2] == 2:
                symbol = '<'
            key = 'x{} {} {:.3f}'.format((item[0]-3), symbol, item[1][0])
        return key

    def rank_sum(dic, topN_score):
        topN_score.reverse()
        for i in range(0, len(topN_score)):
            key = topN_score[i]

            if key in dic:
                dic[key] = dic[key] + (i+1)
            else:
                dic[key] = (i+1)
        return dic

    def print_all(save_all):
        print('rule \t\t ratio \t   minus    trt_effec  nongroup_mean   number      p_val  non_p_val')
        for i in save_all:
            ratio, minus, trt_effect,nongroup_mean, number, p_val, non_p_val = i[1]
            print('{:12s}: {:10.3f} {:10.3f}  {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f}'.format(return_key(i[0]), ratio, minus, trt_effect,nongroup_mean, number, p_val, non_p_val))

    def write(topN_score):
        for item in topN_score:
            key = return_key(item[2])
            print('{} : {:.5f}'.format(key, item[1]))
            
    def sss(all_score, attribute, ascend):
        topN = all_score.sort_values(by = attribute, ascending=ascend).head(10)
        li = topN['rule'].values.tolist()
        return li

    def analysis(t):
        df, start_dataset, end_dataset, rule_len = t
        
        # Set default group number to every patients
        if 'group' not in df:
            df['group'] = 0
        
           
        for did in range(start_dataset, end_dataset+1):
            
            #initial
            idx = ['rule', 'ratio', 'minus', 'trt_effect', 'nongroup_trt_effect', 'number', 'p_val', 'non_p_val', 'abs(ratio)', 'abs(number-100)']
            all_score = pandas.DataFrame([], columns=(idx))
            save_all = []
            r_dic = {}
            rank = []
            
            print('Dataset {}'.format(did))
            d = df[df['dataset'] == did].copy()
            
            for rule in possible_rules(d):
                d = df[df['dataset'] == did].copy()
                
                for i in range(1, 241):
                    patient = d[d['id'] == i].iloc[0] 
                    g = check_group(patient, rule)
                    d.loc[d['id'] == i, 'group'] = g
                
                # Evaluate the treatment effect according each rule
                parts = evaluation(d)
                if type(parts) == int:
                    continue

                # Caculate the socre
                ratio, minus, trt_effect, nongroup_trt_effect, number, p_val, non_p_val = parts
                save_all.append([return_key(rule), ratio, minus, trt_effect, nongroup_trt_effect, number, p_val, non_p_val, abs(ratio), abs(number-100)])
                
            for i in range(0, len(save_all)):
                all_score.loc[i] = save_all[i] 
            all_score.to_csv('Dataset{}_score.csv'.format(did))   

            
            #topN_Ratio = sss(all_score, 'abs(ratio)', False )
            topN_Minus = sss(all_score, 'minus', True )
            topN_TreatEffect = sss(all_score, 'trt_effect', True )
            topN_Nongroup_TreatEffect = sss(all_score, 'nongroup_trt_effect', False )
     
            #r_dic = rank_sum(r_dic, topN_Ratio.copy())
            r_dic = rank_sum(r_dic, topN_Minus.copy())
            r_dic = rank_sum(r_dic, topN_TreatEffect.copy())
            r_dic = rank_sum(r_dic, topN_Nongroup_TreatEffect.copy())
            
            for k, v in r_dic.items():
                rank.append((k, v))
            rank = sorted(rank, key=lambda a:a[1], reverse=True)
            
            print('Rank Sum (Rule, score)')
            for i in range(0, len(rank)):
                print(rank[i])
            print('\n')
