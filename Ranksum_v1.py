
# coding: utf-8

# In[ ]:




# In[101]:

import time
import pandas
import statistics
import threading
import _thread
import numpy as np
import zipfile
import analysis as a
from scipy import stats  



#data load
testing_data = pandas.read_csv("dataset/Data.csv")
training_data = pandas.read_csv("dataset/Training_Data.csv")
training_truth = pandas.read_csv("dataset/Training_Data_truth.csv")


# In[13]:

def caculation(data):
    trt_1 = data[data['trt'] == 1]
    trt_0 = data[data['trt'] == 0]
    
    medi = statistics.median(trt_1['y']) - statistics.median(trt_0['y'])
    mean = statistics.mean(trt_1['y']) - statistics.mean(trt_0['y'])
    peop = len(trt_1) + len(trt_0)
    vari = statistics.variance(trt_1['y']) + statistics.variance(trt_0['y'])
    z_stat, p_val = stats.ranksums(trt_0['y'], trt_1['y']) 
    return [medi, mean, peop, p_val]

def dataset_2():
    
    print('dataset_{}\n{:<15s} {:<15s} {:<15s} {:<15s} {:<15s}'.format("2","Rule", "Median", "Mean", "#ofpeople", "P_val"))
    
    rang = ((51.1,), (51.2,), (51.3,), (51.4,), (52,), (55,), (60,))
    data = training_data[training_data['dataset'] == 2]

    for a in rang:
        data_in = data[data['x29']< a[0]]
        data_out = data[data['x29']>= a[0]]

        w(data_in, data_out, a)
        
    print('\n')
       
    
def dataset_3():
    
    print('dataset_{}\n{:<15s} {:<15s} {:<15s} {:<15s} {:<15s}'.format("3","Rule", "Median", "Mean", "#ofpeople", "P_val"))
    
    rang = [(1, 2), (0, 2), (0, 1)]
    data = training_data[training_data['dataset'] == 3]
    for a in rang:
        data_in = data[((data['x5'] == a[0]) | (data['x5'] == a[1]))]
        data_out = data[((data['x5'] != a[0]) & (data['x5'] != a[1]))]
        
        w(data_in, data_out, a)
    print('\n')

def dataset_4():
    
    #print('dataset_{}\n{:<20s} {:<20s} {:<20s} {:<20s} {:<20s} {:<20s}'.format("4","Rule", "Median", "Mean", "#ofpeople", "P_val", "P_val_o"))
    print('dataset_{} {:<20s} {:<20s} {:<20s} {:<20s} {:<20s} {:<20s}'.format("4","Rule", "Mean", "Mean_in","Mean_out", "people_in", "P_val", "P_val_o"))
    
    rang = [(1, 2, 55), (1, 2, 57.7), (1, 2, 58), (1, 2, 60)]
    data = training_data[training_data['dataset'] == 4]
    for a in rang:
        subgroup = data[((data['x4'] == a[0]) | (data['x4'] == a[1])) & (data['x22'] > a[2])]
        notsubgroup = data[((data['x4'] != a[0]) & (data['x4'] != a[1])) | (data['x22'] <= a[2])]
       
        w(data_in, data_out, a)
        print('\n')
   
    
def te():
    
    print('dataset_{} \n{:<15s} {:<15s} {:<15s} {:<15s} {:<15s} {:<15s}'.format("t","Rule", "mean_in", "ratio", "P_val", "p_val_out", "people"))
        

    d = training_data[training_data['dataset'] == 2]
    subgroup = d[(d['x29']  < 51.3)]
    nsg= d[(d['x29'] >= 51.3 ) ]
     
    a = w(subgroup, nsg, ('x14', 0))
    
    #print(str(a) +' ' + str(b) + ' '+ str(a*b))
    print('\n')

def w(data_in, data_out, i):

    [median_in, mean_in, people_in, p_val_in] = caculation(data_in) #selection rules
    [median_out, mean_out, people_out, p_val_out] = caculation(data_out)
    
    #output
    median = median_in / median_out
    ratio = mean_in / mean_out
    minus = mean_in - mean_out
    people = people_in
    p_val = p_val_in 
    if type(i) != tuple:
        i = tuple(i)
    rule ='(' + str(i[0])
    for pos in range(1, len(i)):
        rule = rule + ', ' + str(i[pos])
    rule = rule + ')'

    print('{:<15s} {:<15.5f} {:<15.5f} {:<15.5f} {:<15.5f} {:<15.5f}'.format(rule, mean_in, ratio, p_val, p_val_out, people ))
    
    return ratio
     
#dataset_2()
#dataset_3()
#dataset_4()
te()


# In[37]:

#make random 
f = open('output/rands.csv', 'w')
f.write("id")
for i in range(1,1200):
    f.write(",dataset_" + str(i))
f.write("\n")
for i in range(1, 241):
    f.write(str(i))
    for j in range(1, 1200):
        f.write("," + str(random.randint(0,1)))
    f.write("\n")
f.close()

#make all ones
f = open('output/one.csv', 'w')
f.write("id")
for i in range(1,1200):
    f.write(",dataset_" + str(i))
f.write("\n")
for i in range(1, 241):
    f.write(str(i))
    for j in range(1, 1200):
        f.write(",1")
    f.write("\n")
f.close()
    


# In[223]:


data = f[f['dataset'] == 1]
trt1 = data[data['trt'] == 1]
trt0 = data[data['trt'] == 0]
trt1MinN = trt1.sort(['y'], ascending = 1)
trt0MaxN = trt0.sort(['y'], ascending = 0)
statistics.mean(trt0MaxN.head()['y']) - statistics.mean(trt1MinN.head()['y'])

a=100

for i in range(0, trt0MaxN.index.size):
    for j in range(0, trt1MinN.index.size):
        t = statistics.mean(trt0MaxN.head(5)['y']) - statistics.mean(trt1MinN.head(5)['y'])
        if(t<a && t>60)
    

#後面的數字會越來越大, 前面的數字會越來越小


# In[370]:

#SVM training file
trainData = f1.copy()
trainAns = f2.copy()
file = open('dataset/svmtrain', 'w')
for i in range(1, 5):
    tw = trainData[trainData['dataset'] == i]
    aw = trainAns[('dataset_'+ str(i))]
    s = ""
    for j in range(0, 240):
        s = s + str(aw[j])
        for k in range(1, 43):
            s = s + " " + str(k) + ":" + str(tw.loc[240*(i-1) + j][k+1])
        s = s + "\n"
    file.write(s)
file.close


# In[1]:

import pandas
import threading
import _thread
import time

def Threadfun(sleeptime, start, end):
    s = ""
    for i in range(start, end): #of dataset
        tw = testData[testData['dataset'] == i]
        for j in range(0, 240): #of id
            s = s + "0"
            for k in range(1, 43):    #of feature
                s = s + " " + str(k) + ":" + str(tw.loc[240*(i-1) + j][k+1])
            s = s + "\n"
        time.sleep(sleeptime)
        print('thread_{0}: {1}'.format(start, i))
    file = open(('dataset/svmtest_{0}'.format(start)), 'w')

    file.write(s)
    file.close 

def main():

    # _thread.start_new_thread(Threadfun, (0.1, 1, 241))
    # _thread.start_new_thread(Threadfun, (0.1, 600, 720))
    _thread.start_new_thread(Threadfun, (0.1, 720, 850))
    _thread.start_new_thread(Threadfun, (0.3, 850, 960))
    _thread.start_new_thread(Threadfun, (0.5, 960, 1100))
    _thread.start_new_thread(Threadfun, (0.7, 1100, 1201))

#test file    

testData = testing_data.copy()

main()



# In[21]:

#make svm csv

svm = open('dataset/svmtest.predict', 'r')
ss = svm.read().splitlines()
svm.close()
ans = pandas.DataFrame([], columns=(['id'] + ['dataset_{}'.format(i) for i in range(1, 1201)]))
for col in range(1, 1201):
    for row in range(1, 241):
        ans.loc[row , 'dataset_'+str(colu)] = ss[240*(col-1)+row][0]
for row in range(1, 241):
    ans.loc[row, 'id'] = row
ans.to_csv('svm.csv')      


# In[21]:

#make upload file

dataset_n = 7
feature = 'x14'
value = (1, 0)

output_file = 'dataset{}_{}.csv'.format(dataset_n, feature)

ans = pandas.DataFrame([], columns=(['id'] + ['dataset_{}'.format(i) for i in range(1, 1201)]))
ans['id'] = range(1, 241)

d = test[test['dataset'] == dataset_n]
if(len(value)== 1):
    subgroup = d[(d[feature] == value[0])]
else:
    subgroup = d[(d[feature] == value[0]) | (d[feature] == value[1])]


for col in range(1, 1201):
    ans['dataset_{}'.format(col)] = 0
    if(col == dataset_n):
        for i in subgroup['id']:
            ans.loc[i-1, 'dataset_{}'.format(col)] = 1
        
ans.to_csv(output_file)   
zf = zipfile.ZipFile('result_bf_{}.zip'.format(name), 'w', zipfile.ZIP_DEFLATED)
    zf.write('result_bf_{}.csv'.format(name))


# In[97]:

# Configuration
P_VALUE_THRESHOLD = 0.05
TRT_EFFECT_THRESHOLD = -0.6

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


# In[100]:

params = (training_data, 4, 4, 1)

start_time = time.time()
s_all = analysis(params)
print("\n--- {} seconds ---\n".format(time.time() - start_time))
#跑一個dataset大概104s


# In[ ]:

def parallel_execution():
    # Engine initialization
    c = Client()
    v = c[:]
    v.block = True
    
    # Load dataset
    testing_data = pandas.read_csv("dataset/Data.csv")
    
    # Run brute-force parallelly
    # Parameters: Dataframe, Start Dataset, End Dataset, Rule length
    # Return: Subgroup, Best Rules
    params = []
    for i in range(8):
        params.append((testing_data, (i*150)+1, (i+1)*150, 1))
    return v.map(b.brute_force, params)

# Calculate the execution time
start_time = time.time()
results = parallel_execution()
print("\n--- {} seconds ---\n".format(time.time() - start_time))


# In[ ]:



