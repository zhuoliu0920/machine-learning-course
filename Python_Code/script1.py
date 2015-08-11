#!/usr/bin/env python3

""" Generating bar plot for the counts of each classes, and histograms for each predictor on each class.
"""

__author__ = "Zhuo Liu"

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

# Generate header as a list of strings
with open('segmentation.data', 'r') as f:
    header_line = str(f.readlines()[3])
    header_list = ['CLASS'] + header_line.strip().split(',')

# Read data files as pandas data frame
df = pd.read_csv('segmentation.data', header=None, skiprows=4) # skip the first 4 rows which are the description and header of the data set
df2 = pd.read_csv('segmentation.test', header=None, skiprows=4) # skip the first 4 rows which are the description and header of the data set
df = df.append(df2, ignore_index=True)
df.columns = header_list # assign column names (headers)

# Get bar plot for counts by classes
fig1 = plt.figure(figsize=(16.0, 9.0))
df_classes = df.ix[:, 0]
df_classes.value_counts().plot(kind='bar')

fig1.savefig('./Plots/1_BarbyClasses.png')

# Get histograms for each predictors on different classes
classes = set(df.ix[:, 0])
fig2=[]
for i in range(19):
    fig = plt.figure(figsize=(9.0, 16.0))
    for j,c in enumerate(classes):
        plt.subplot(4,2,j+1) 
        df_sub = df.iloc[:, i+1][df.iloc[:,0]==c]
        df_sub.hist()
        plt.title('Histogram for ' + c)
        plt.xlabel(df.columns[i+1])
        plt.ylabel('frequency')
    fig.tight_layout()
    fig2.append(fig)
    fig2[i].savefig('./Plots/1_'+str(i)+'Hist.png')
        


