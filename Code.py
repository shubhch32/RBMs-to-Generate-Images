# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 23:04:38 2019

@author: ShubhamCh
"""

#train.csv and test.csv must be in the same folder.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from scipy.stats import bernoulli
from sklearn.manifold import TSNE
np.random.seed(1234)
train = "train.csv"
test = "test.csv"
train = pd.read_csv(train)
test = pd.read_csv(test)
train = train.drop('id',axis=1)
test = test.drop('id',axis=1)

#Preprocessing
x_train = np.asarray(train.loc[:, train.columns != 'label'])
x_test = np.asarray(test.loc[:, test.columns != 'label'])
y_train = np.asarray(train['label'])
y_test = np.asarray(test['label'])
x_train[x_train<127]=0
x_train[x_train>=127]=1
x_test[x_test<127]=0
x_test[x_test>=127]=1

def sigmoid(x):
    return 1/(1+np.exp(-x))

n=100
m=784
lr=0.1
epochs=1
k=20
w=np.random.randn(n,m)*np.sqrt(2.0/(x_train.shape[0]))
b=np.random.randn(m)*np.sqrt(2.0/(x_train.shape[0]))
c=np.random.randn(n)*np.sqrt(2.0/(x_train.shape[0]))
loss=[]
ncols = 9
nrows = 9

fig = plt.figure()
axes = [ fig.add_subplot(nrows, ncols, r * ncols + c) for r in range(1, nrows) for c in range(1, ncols) ]
ii=0

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

for epoch in range(epochs):
    for i in range(x_train.shape[0]):
        v=x_train[i]
        for t in range(k):
            z_h=sigmoid(np.dot(w,v)+c)
            h=bernoulli.rvs(size=n,p=z_h)
            z_v=sigmoid(np.dot(np.transpose(w),h)+b)
            v=bernoulli.rvs(size=m,p=z_v)
        w=w+lr*(np.dot(np.reshape(sigmoid(np.dot(w,x_train[i])+c),[n,1]),np.reshape(x_train[i],[1,m]))-np.dot(np.reshape(sigmoid(np.dot(w,v)+c),[n,1]),np.reshape(v,[1,m])))
        b=b+lr*(x_train[i]-v)
        c=c+lr*(sigmoid(np.dot(w,x_train[i])+c)-sigmoid(np.dot(w,v)+c))
        v=x_train[i]
        #finding reconstruction error
        for t in range(2):
            z_h=sigmoid(np.dot(w,v)+c)
            h=bernoulli.rvs(size=n,p=z_h)
            z_v=sigmoid(np.dot(np.transpose(w),h)+b)
            v=bernoulli.rvs(size=m,p=z_v)
        loss.append(sum((v-x_train[i])**2))
        if i%100==0 and ii<64:
            axes[ii].imshow(np.reshape(v,[28,28]))
            ii+=1

plt.show()

#Reconstruction moving average loss
loss1=[]
s=0.0
window=100
for i in range(window):
    s=s+loss[i]
for i in range(window,len(loss)):
    loss1.append(s/window)
    s=s+loss[i]-loss[i-window]
plt.plot(np.array(loss1))
plt.show()

#T-SNE plot
z_h=np.transpose(sigmoid(np.dot(w,np.transpose(x_test))+np.reshape(c,[n,1])))
h=bernoulli.rvs(size=[x_test.shape[0],n],p=z_h)
tsne=TSNE(n_components=2).fit_transform(h)

import matplotlib
fig = plt.figure(figsize=(8,8))
color=['red','green','blue','purple','pink','brown','black','orange','yellow','cyan']
plt.scatter(tsne[:,0], tsne[:,1], c=y_test,cmap=matplotlib.colors.ListedColormap(color),alpha=0.8)
# plt.scatter(tsne[:,0], tsne[:,1], c=y_test)
cb = plt.colorbar()
loc = np.arange(0,max(y_test),max(y_test)/float(10))
cb.set_ticks(loc)
cb.set_alpha(0.5)
cb.set_ticklabels(color)
plt.show()