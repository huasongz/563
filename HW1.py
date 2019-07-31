#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:50:17 2019

@author: huasongzhang
"""

from mnist import MNIST
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import cvxpy as cp

# load data
def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train, labels_train, X_test, labels_test

X_train, Y_tr, X_test, Y_te = load_dataset()
Y_train = np.zeros((60000,10))
Y_train[np.arange(60000), Y_tr] = 1
Y_test = np.zeros((10000,10))
Y_test[np.arange(10000), Y_te] = 1
m = X_train.shape[1]
n = Y_train.shape[1]

method = ['pinv','LASSO','Ridge']



# use various methods to solve this problem
# pinv
x1 = np.dot(np.linalg.pinv(X_train), Y_train)

# LASSO
clf = linear_model.Lasso(alpha=0.005)
clf.fit(X_train, Y_train)
x2 = clf.coef_
x2 = np.transpose(x2)

# Ridge
clf1 = Ridge(alpha=0.005)
clf1.fit(X_train, Y_train) 
x3 = clf1.coef_
x3 = np.transpose(x3)






# Analysis on the overall coefficients
sum_x1 = np.sum(abs(x1),axis = 1)
plt.bar(np.arange(m), sum_x1, alpha=0.5)
plt.xlabel('Pixel')
plt.ylabel('Coefficient')
plt.title('Absolute value of each pixel using pinv method')
plt.savefig('coeff_pinv.jpg')
sum_x2 = np.sum(abs(x2),axis = 1)
plt.bar(np.arange(m), sum_x2)
plt.xlabel('Pixel')
plt.ylabel('Coefficient')
plt.title('Absolute value of each pixel using LASSO method')
plt.savefig('coeff_LASSO.jpg')
sum_x3 = np.sum(abs(x3),axis = 1)
plt.bar(np.arange(m), sum_x3, alpha=0.5)
plt.xlabel('Pixel')
plt.ylabel('Coefficient')
plt.title('Absolute value of each pixel using Ridge method')
plt.savefig('coeff_Ridge.jpg')




# find out the accuracy
for i in range(3):
    count = 0
    X = globals()["x" + str(i+1)]
    globals()['Y'+str(i+1)] = np.dot(X_test,X)
    globals()['Y'+str(i+1)] = np.argmax(globals()['Y'+str(i+1)], axis=1)
    for j in range(len(globals()['Y'+str(i+1)])):
        if globals()['Y'+str(i+1)][j] == Y_te[j]:
            count += 1
    accuracy = count/len(globals()['Y'+str(i+1)])
    print('The accuracy by using',method[i],'is',accuracy)    
    


# find out the most informative pixels sum_xi_d for i = 1:5
R = np.array([50,100,300])
for j in range(3):
    r = R[j] #fix r
    a = np.ones((10,784))
    for i in range(3):
        x = np.zeros(784,)
        xx = globals()['sum_x'+str(i+1)]
        index = xx.argsort()[-r:][::-1]
        x[index] = 1
        x_index = np.transpose(a*x)
        x_reduced = globals()['x'+str(i+1)]*x_index
        # test accuracy
        pre = np.dot(X_test,x_reduced)
        l = len(pre)
        count = 0
        for h in range(l):
            max_value = max(pre[h,])
            max_index = np.where(pre[h,] == max_value)
            if max_index[0][0] == Y_te[h]:
                count += 1
        accuracy = count/l
        print('Accuracy with rank = ',R[j],'for method',method[i],'is', accuracy)




r = 300
# Analysis on the coefficient for each digit
for j in range(3):
    x = globals()['x'+str(j+1)]
    for i in range(10):
        x_initial = np.zeros(784,)
        x_digit = x[:,i]
        index = x_digit.argsort()[-r:][::-1]
        x_initial[index] = 1
        x[:,i] = x_digit*x_initial
    globals()['x_digit_reduced_'+str(j+1)] = x

# calculate the accuracy
for i in range(3):
    pre = np.dot(X_test,globals()['x_digit_reduced_'+str(i+1)])
    count = 0
    l = len(pre)
    for h in range(l):
        max_value = max(pre[h,])
        max_index = np.where(pre[h,] == max_value)
        if max_index[0][0] == Y_te[h]:
            count += 1
    accuracy = count/l
    print('Accuracy with rank = 300','for method',method[i],'is', accuracy)


fig, ax = plt.subplots(2,5,figsize=(20, 8))
for i in range(10):
    ax = plt.subplot(2,5,i+1)
    size = np.reshape(x_digit_reduced_2[:,i],(28,28))
    plt.pcolor(size)   
    ax.set_title(i)  
fig.suptitle('Informative pixels for each digit')
fig.savefig('digit.jpg')

        
        
        