import os
import pandas as pd
import torch 
import numpy as np
from scipy import stats
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


train_file_path = '/data/zhanglab/lli1/methylation/train_combat.csv'
test_file_path = '/data/zhanglab/lli1/methylation/test_combat.csv'
corr_path = '../../data/correlation.csv'


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x_in):
        return self.linear(x_in)

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

    def init_weights(self):
        init.xavier_normal(self.fc1.weight) 
        init.xavier_normal(self.fc2.weight)
        
    def forward(self, x_in, apply_softmax=False):
        a_1 = F.leaky_relu(self.fc1(x_in))
        y_pred = F.leaky_relu(self.fc2(a_1))
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)

        return y_pred


class Trainer(object):
    """trains and tests the model"""
    def __init__(self,epoch,model,batch_size):
        self.model = model
        self.epoch = epoch
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()
        self.batch_size = batch_size

    def train_by_random(self, train):
        for t in range(0,self.epoch):
            from math import ceil
            loss = 0
            correct = 0
            acc = 0
            random_index = torch.randperm(len(train))
            random_train = train[random_index]

            x_train = random_train[:,1:]
            y_train = random_train[:,0].reshape(-1,1)
            for i in range(0,ceil(len(x_train) // self.batch_size)):
                start_index = i*self.batch_size
                end_index = (i+1)*self.batch_size if (i+1)*self.batch_size <= len(x_train) else len(x_train)

                random_train = x_train[start_index:end_index,:]
                y_labels = y_train[start_index:end_index].reshape(-1,1)
                y_pred = self.model(random_train)
                acc +=  torch.sum(torch.abs(torch.sub(y_labels, y_pred)))
                
                loss = self.loss_fn(y_pred, y_labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if (t%20==0):
                print ("epoch: {0:02d} | loss: {1:.2f} | acc: {2:.2f}".format(t, loss, acc / len(y_train)))

    @staticmethod
    def get_accuracy(y_labels,y_pred):
        difference = torch.abs(y_labels-y_pred)
        correct = sum(list(map(lambda x: 1 if x<2 else 0,difference)))
        return correct


    def test(self,x_test,y_test):
        model = self.model.eval()
        pred_test = model(x_test)
        
        x_arr = torch.abs(torch.sub(pred_test,y_test))
        x_arr = x_arr.cpu().data.numpy()
        x_sz = len(x_arr)
        mae = np.sum(x_arr)/x_sz
        
        rss = torch.sum(torch.sub(pred_test, y_test) ** 2)
        y_mean = torch.mean(y_test)
        tss = torch.sum(torch.sub(y_test, y_mean) ** 2)
        rsquared = 1-rss/tss
        
        print ("MAE = {} R^2 = {}".format(mae, rsquared))

        return mae, rsquared


def calculate_correlation(train):
    """Caculates the spearman correlation each feature and the label.  

    Args: 
        train: The training data. A 2D array where rows are data points, the first column is 
        the labels, and the rest are feature values.

    Returns:
        A 1D array containing the spearman correlation of each data point in the training data.
    """
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
    spearman_indices.insert(0,0)
    return spearman_indices
    
def slice_data(data):
    """ Splits a dataset into columns 2+ (features) and column 1 (labels)."""
    features = data[:,1:]
    labels = data[:,:1]
    return features, labels

train = load_training_data()
test  = load_testing_data()

def train_and_test_NN(input_dim, hidden_dim, epochs, batch_size):#TODO add multiple layers
    indices = get_filtered_indices(input_dim)
    sub_train = train[:,indices]
    sub_test = test[:,indices]
    sub_train = torch.from_numpy(sub_train).float()
    sub_test = torch.from_numpy(sub_test).float()
    x_test, y_test = slice_data(sub_test)
    model = NeuralNet(input_dim,hidden_dim,1)
    trainer = Trainer(epochs,model,batch_size)
    trainer.train_by_random(sub_train)
    mae, r2 = trainer.test(x_test,y_test)
    return model, mae, r2

def train_and_test_Linear(input_dim, epochs, batch_size):
    indices = get_filtered_indices(input_dim)
    sub_train = train[:,indices]
    sub_test = test[:,indices]
    sub_train = torch.from_numpy(sub_train).float()
    sub_test = torch.from_numpy(sub_test).float()
    x_test, y_test = slice_data(sub_test)
    model = LinearRegression(input_dim,1)
    trainer = Trainer(epochs,model,batch_size)
    trainer.train_by_random(sub_train)
    mae, r2 = trainer.test(x_test,y_test)
    return model, mae, r2
