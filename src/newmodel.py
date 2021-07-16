import pandas as pd
import torch 
import numpy as np
from scipy import stats
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from sklearn import linear_model

train_file_path = '/data/zhanglab/lli1/methylation/train_combat.csv'
test_file_path = '/data/zhanglab/lli1/methylation/test_combat.csv'
corr_path = '../data/correlation.csv'


def calculate_correlation(train):
    print("Calculating feature correlation....")
    start = time.time()
    spearman_corr = []
    for i in range(1, train.shape[1]):
        spearman_corr.append(stats.spearmanr(train[:,0],train[:,i])[0])
    end = time.time()
    print("Done calculating correlation. Time (min) = ", (end - start)/60)
    spearman_corr = np.array(spearman_corr)
    np.savetxt(corr_path, spearman_corr, delimiter = ',')
    return spearman_corr

def load_training_data():
    print("Loading training data...")
    start = time.time()
    train = np.loadtxt(train_file_path, delimiter=',')
    print("shape = ", train.shape)
    end = time.time()
    print("Loaded training data. Time (min) = ", (end-start)/60)
    return train

def load_testing_data():
    print("Loading testing data...")
    start = time.time()
    test = np.loadtxt(test_file_path, skiprows=1, delimiter=',')
    print("shape = ", test.shape)
    end = time.time()
    print("Loaded testing data. Time (min) = ", (end-start)/60)
    return test

def get_filtered_indices(filter_size, recalc_corr = False, corr_path = corr_path, filter_start = 0):
    # Load feature correlation, or calculate it
    spearman_corr = calculate_correlation(train) if recalc_corr else np.loadtxt(corr_path, delimiter = ',')
    sorted_corr = np.sort(abs(spearman_corr))[::-1] # descending
    # filter indices with correlation above cutoff
    start = sorted_corr[filter_start]
    end = sorted_corr[filter_start + filter_size]
    spearman_indices = [x+1 for x in range(len(spearman_corr)) if end < abs(spearman_corr[x]) <= start]
    return spearman_indices
    
filter_size = 100
train = load_training_data()
test  = load_testing_data()
indices = get_filtered_indices(filter_size)
x_train = train[:,indices]
y_train = train[:,0]
x_test = test[:,indices]
y_test = test[:,0]

i = 0
feature_models = linear_model.LinearRegression()
x = x_train[:,i]
feature_models.fit(x,y_train)
