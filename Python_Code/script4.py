#!/usr/bin/env python3

""" Building classifiers by SVM with different kernals (linear, gaussian and polynomial). Find the optimal parameters.
"""

__author__ = "Zhuo Liu"

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC

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

# Apply PCA
pca = PCA() # optimal sigma=35.355, gamma = 1/(2*sigma^2)
pc_data = pca.fit_transform(df.iloc[:,1:])

# Apply SVM classifier with linear kernel
X = df.iloc[:,1:] # predictors
y = df.iloc[:,0] # responses
svc = SVC(kernel='linear', C=10).fit(X, y) # C = 10 is optimal
rbf_svc = SVC(kernel='rbf', gamma=0.0001, C=20).fit(X, y)
ploy_svc = SVC(kernel='poly', degree=2, C=1.2).fit(X, y)

# color for different classes
color_dict = { 'BRICKFACE':'red', 'FOLIAGE':'blue', 'GRASS':'black', 'CEMENT':'green','SKY':'brown', 'WINDOW':'orange', 'PATH':'grey' }

#...................................................................................................
# Generate scatter plot on the first 2 PC space (linear kernel)
fig1 = plt.figure(figsize=(16.0, 9.0))
# shape for support vectors is 'x', the others are '.'
marker_list = ['.']*(df.shape[0])
for pos in svc.support_:
    marker_list[pos] = 'x'

for i in range(df.shape[0]):
    plt.scatter( x=pc_data[i,0], y=pc_data[i,1], marker=marker_list[i], c=color_dict[y[i]] )
plt.title('Projection onto first 2 PC space (SVM with linear kernel)')
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
fig1.savefig('./Plots/4_LinearSVM.png')

#...................................................................................................
# Generate scatter plot on the first 2 PC space (rbf kernel)
fig2 = plt.figure(figsize=(16.0, 9.0))
# shape for support vectors is 'x', the others are '.'
marker_list = ['.']*(df.shape[0])
for pos in rbf_svc.support_:
    marker_list[pos] = 'x'

for i in range(df.shape[0]):
    plt.scatter( x=pc_data[i,0], y=pc_data[i,1], marker=marker_list[i], c=color_dict[y[i]] )
plt.title('Projection onto first 2 PC space (SVM with rbf kernel)')
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
fig2.savefig('./Plots/4_RbfSVM.png')

#...................................................................................................
# Generate scatter plot on the first 2 PC space (polynomial kernel)
fig3 = plt.figure(figsize=(16.0, 9.0))
# shape for support vectors is 'x', the others are '.'
marker_list = ['.']*(df.shape[0])
for pos in ploy_svc.support_:
    marker_list[pos] = 'x'

for i in range(df.shape[0]):
    plt.scatter( x=pc_data[i,0], y=pc_data[i,1], marker=marker_list[i], c=color_dict[y[i]] )
plt.title('Projection onto first 2 PC space (SVM with polynomial kernel)')
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
fig3.savefig('./Plots/4_PolySVM.png')


        





        


