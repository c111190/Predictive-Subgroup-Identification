#test file
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

    _thread.start_new_thread(Threadfun, (0.1, 1, 100))
    _thread.start_new_thread(Threadfun, (0.2, 100, 200))
    _thread.start_new_thread(Threadfun, (0.4, 200, 300))
    _thread.start_new_thread(Threadfun, (0.3, 300, 400))
    _thread.start_new_thread(Threadfun, (0.5, 400, 500))
    _thread.start_new_thread(Threadfun, (0.7, 500, 600))

#test file    
strartime = time.time() 
f = pandas.read_csv('dataset/Data.csv')
testData = f.copy()

main()

print("\nprocessing time: {}\n".format(time.time() - strartime))