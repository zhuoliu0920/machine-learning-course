#!/usr/bin/env python3

""" By applying PCA, project original data space onto first and second 2 principle component spaces. Then, applying LDA on the PC space, project onto first and second linear discriminant spaces.
"""

__author__ = "Zhuo Liu"

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
pca = PCA()
pc_data = pca.fit_transform(df.iloc[:,1:])
print(pca.explained_variance_ratio_)

# Generate scatter plot on the first 2 PC space
color_dict = { 'BRICKFACE':'red', 'FOLIAGE':'blue', 'GRASS':'black', 'CEMENT':'green','SKY':'brown', 'WINDOW':'orange', 'PATH':'grey' }
fig1 = plt.figure(figsize=(16.0, 9.0))
plt.scatter( x=pc_data[:,0], y=pc_data[:,1], c=[color_dict[c] for c in df.iloc[:,0]] )
plt.title('Projection onto first 2 PC space')
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
fig1.savefig('./Plots/2_PCA1.png')

# Generate scatter plot on the second 2 PC spacefig2 = plt.figure()
fig2 = plt.figure(figsize=(16.0, 9.0))
plt.scatter( x=pc_data[:,2], y=pc_data[:,3], c=[color_dict[c] for c in df.iloc[:,0]] )
plt.title('Projection onto second 2 PC space')
plt.xlabel('Principle Component 3')
plt.ylabel('Principle Component 4')
fig2.savefig('./Plots/2_PCA2.png')

# Apply LDA to project PC space onto first 2 LD space
lda = LDA()
ld_data = lda.fit_transform(X=pc_data, y=df.iloc[:,0])
fig3 = plt.figure(figsize=(16.0, 9.0))
plt.scatter( x=ld_data[:,0], y=ld_data[:,1], c=[color_dict[c] for c in df.iloc[:,0]] )
plt.title('Projection onto first 2 LD space')
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
fig3.savefig('./Plots/2_LDA1.png')

# Apply LDA to project PC space onto second 2 LD space
fig4 = plt.figure(figsize=(16.0, 9.0))
plt.scatter( x=ld_data[:,2], y=ld_data[:,3], c=[color_dict[c] for c in df.iloc[:,0]] )
plt.title('Projection onto second 2 LD space')
plt.xlabel('Linear Discriminant 3')
plt.ylabel('Linear Discriminant 4')
fig4.savefig('./Plots/2_LDA2.png')


        


