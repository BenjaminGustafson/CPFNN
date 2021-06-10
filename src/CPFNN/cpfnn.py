import os
import pandas as pd
import torch 
import numpy as np
from scipy import stats
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

class Config(object):
    """stores the global variables"""
    train_file_path = '/data/zhanglab/lli1/methylation/train_combat.csv'
    test_file_path = '/data/zhanglab/lli1/methylation/test_combat.csv'
    corr_path = '../../data/correlation.csv'
    model_path = '../../data/model02.pt'
    filter_size = 3000
    hidden_dim = 200
    epoch = 2000
    batch_size = 50
    features = 473034
    output_dim = 1
    use_gpu = True
    recalc_corr = False
    debug = False
    """
    train_file_path -- path to the training data file
    test_file_path -- path to the testing data file
    filter_size -- number of features that we keep
    hidden_dim -- number of nodes in the hidden layer 
    epoch -- number of epochs of training, i.e. how long to train the model
    batch_size -- 
    features -- number of features before filtering
    use_gpu -- will use GPU if available
    recalc_corr -- recalculate spearman correlation, otherwise load from file
    debug -- read only 2 lines from training and testing data
    """



class NeuralNet(nn.Module):
    """A neural network model with 1 hidden layer"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        input_dim --
        hidden_dim --
        output_dim --
        """
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
            #proint(random_train.shape)
            x_train = random_train[:,1:]
            y_train = random_train[:,0].reshape(-1,1)
            for i in range(0,ceil(len(x_train) // self.batch_size)):
                start_index = i*self.batch_size
                end_index = (i+1)*self.batch_size if (i+1)*self.batch_size <= len(x_train) else len(x_train)
            #    print(start_index,self.batch_size,end_index)
                random_train = x_train[start_index:end_index,:]
                y_labels = y_train[start_index:end_index].reshape(-1,1)
                y_pred = self.model(random_train)
                acc +=  torch.sum(torch.abs(torch.sub(y_labels, y_pred)))
                # Loss
                #penalty = alpha * self.model.corr_l1_penalty + beta * self.model.corr_l2_penalty
                loss = self.loss_fn(y_pred, y_labels)#+penalty
                # Zero all gradients
                self.optimizer.zero_grad()
                #Backward pass
                loss.backward()
                # Update weights
                self.optimizer.step()
                                # Verbose
            if (t%20==0):
                # Print the gredient
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

        print("MAE = " , np.sum(x_arr)/x_sz)

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
    np.savetxt(Config.corr_path, spearman_corr, delimiter = ',')
    return spearman_corr

def load_training_data():
    print("Loading training data...")
    start = time.time()
    skip = 715 if Config.debug else 0
    train = np.loadtxt(Config.train_file_path, skiprows=skip, delimiter=',')
    print("shape = ", train.shape)
    end = time.time()
    print("Loaded training data. Time (min) = ", (end-start)/60)
    return train

def load_testing_data():
    print("Loading testing data...")
    start = time.time()
    skip = 306 if Config.debug else 1
    test = np.loadtxt(Config.test_file_path, skiprows=skip, delimiter=',')
    print("shape = ", test.shape)
    end = time.time()
    print("Loaded testing data. Time (min) = ", (end-start)/60)
    return test

def get_filtered_indices(filter_size = Config.filter_size, recalc_corr = Config.recalc_corr, corr_path = Config.corr_path, filter_start = 0):
    # Load feature correlation, or calculate it
    spearman_corr = calculate_correlation(train) if recalc_corr else np.loadtxt(corr_path, delimiter = ',')

    sorted_corr = np.sort(abs(spearman_corr))[::-1] # descending

    # filter indices with correlation above cutoff
    start = sorted_corr[filter_start]
    end = sorted_corr[filter_start + filter_size]
    spearman_indices = [x+1 for x in range(len(spearman_corr)) if end < abs(spearman_corr[x]) < start]
    spearman_indices.insert(0,0)
    return spearman_indices

def filter_data(data, indices):
    return data[:,indices]
    
def slice_data(data):
    """ Splits a dataset into columns 2+ (features) and column 1 (labels)."""
    features = data[:,1:]
    labels = data[:,:1]
    return features, labels


#Code run when executed, but not when imported
if __name__ == "__main__":

    # Use GPU or CPU

    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")

    device = None
    if Config.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')  
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print("using GPU")
    else: 
        device = torch.device('cpu')
        print("using CPU")

    # Load data

    train = load_training_data()
    test  = load_testing_data()    

    filtered_indices = get_filtered_indices()

    #filter data

    sub_train = filter_data(train, filtered_indices)
    sub_test = filter_data(test, filtered_indices)
    sub_train = torch.from_numpy(sub_train).float().to(device)
    sub_test = torch.from_numpy(sub_test).float().to(device)

    x_test, y_test = slice_data(sub_test)

    # Train model

    model = NeuralNet(input_dim=Config.filter_size,hidden_dim=Config.hidden_dim,output_dim=Config.output_dim).to(device)
    trainer = Trainer(epoch=Config.epoch,model=model,batch_size=Config.batch_size)

    print("Training model...")
    start = time.time()
    trainer.train_by_random(sub_train)
    end = time.time()
    print("Done training. Time (min) = ", (end-start)/60)

    # Test

    trainer.test(x_test, y_test)

    # Output model
     
    torch.save(model.state_dict(), Config.model_path)

    # Output prediction
    #output = model(x_test).data.numpy()
    #np.savetxt('CPFNN_prediction1.txt', output,delimiter = ',')
