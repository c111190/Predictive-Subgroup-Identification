import numpy as np
import pandas
import matplotlib.pyplot as plt

training_data = pandas.read_csv('dataset/Training_Data.csv')
df = training_data[training_data['dataset'] == 1].copy()
data = []
data.append(df['y'].tolist())
data.append(df[df['trt']==0]['y'].tolist())
data.append(df[df['trt']==1]['y'].tolist())
fs = 10  # fontsize

# demonstrate how to toggle the display of different elements:
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.boxplot(data, labels=['all', 'trt_0', 'trt_1'], showmeans=True)
axes.set_title('Default', fontsize=fs)



fig.suptitle("Dataset 1 boxplot of [y]")
fig.subplots_adjust(hspace=0.4)
plt.show()

