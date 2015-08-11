#!/usr/bin/env python3

""" Using kernal PCA, then LDA. Gaussian (rbf) kernal is used.
"""

__author__ = "Zhuo Liu"

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.lda import LDA

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
kpca = KernelPCA(kernel='rbf', gamma=1/2/(35.355**2)) # optimal sigma=35.355, gamma = 1/(2*sigma^2)
kpc_data = kpca.fit_transform(df.iloc[:,1:])

# Generate scatter plot on the first 2 PC space
color_dict = { 'BRICKFACE':'red', 'FOLIAGE':'blue', 'GRASS':'black', 'CEMENT':'green','SKY':'brown', 'WINDOW':'orange', 'PATH':'grey' }
fig1 = plt.figure(figsize=(16.0, 9.0))
plt.scatter( x=kpc_data[:,0], y=kpc_data[:,1], c=[color_dict[c] for c in df.iloc[:,0]] )
plt.title('Projection onto first 2 Kernel PC space')
plt.xlabel('Kernel Principle Component 1')
plt.ylabel('Kernel Principle Component 2')
fig1.savefig('./Plots/3_KPCA1.png')

# Generate scatter plot on the second 2 PC space
color_dict = { 'BRICKFACE':'red', 'FOLIAGE':'blue', 'GRASS':'black', 'CEMENT':'green','SKY':'brown', 'WINDOW':'orange', 'PATH':'grey' }
fig2 = plt.figure(figsize=(16.0, 9.0))
plt.scatter( x=kpc_data[:,2], y=kpc_data[:,3], c=[color_dict[c] for c in df.iloc[:,0]] )
plt.title('Projection onto second 2 Kernel PC space')
plt.xlabel('Kernel Principle Component 3')
plt.ylabel('Kernel Principle Component 4')
fig2.savefig('./Plots/3_KPCA2.png')

# Apply LDA to project KPC space onto first 2 LD space
lda = LDA()
ld_data = lda.fit_transform(X=kpc_data, y=df.iloc[:,0])
fig3 = plt.figure(figsize=(16.0, 9.0))
plt.scatter( x=ld_data[:,0], y=ld_data[:,1], c=[color_dict[c] for c in df.iloc[:,0]] )
plt.title('Projection of KPC space onto first 2 LD space')
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
fig3.savefig('./Plots/3_KLDA1.png')

# Apply LDA to project KPC space onto second 2 LD space
fig4 = plt.figure(figsize=(16.0, 9.0))
plt.scatter( x=ld_data[:,2], y=ld_data[:,3], c=[color_dict[c] for c in df.iloc[:,0]] )
plt.title('Projection of KPC space onto second 2 LD space')
plt.xlabel('Linear Discriminant 3')
plt.ylabel('Linear Discriminant 4')
fig4.savefig('./Plots/3_KLDA2.png')


        





        


