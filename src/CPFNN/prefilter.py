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
    corr_path = '/data/zhanglab/bgustafs/InterpretableML/data/correlation.csv'
    model_path = '/data/zhanglab/bgustafs/InterpretableML/data/model.pt'
    filter_size = 20000
    hidden_dim = 200
    epoch = 1000
    batch_size = 50
    features = 473034
    output_dim = 1
    use_gpu = True
    recalc_corr = True
    debug = True
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
    print("Calculating feature correlation....")
    start = time.time()
    spearman_corr = []
    for i in range(1, train.shape[1]):
        spearman_corr.append(stats.spearmanr(train[:,0],train[:,i])[0])
    end = time.time()
    print("Done calculating correlation. Time (min) = ", (end - start)/60)

    spearman_corr = np.array(spearman_corr)
    np.savetxt(Config.corr_path, spearman_corr, delimiter = ',')

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

    print("Loading training data...")
    start = time.time()
    skip = 705 if Config.debug else 0
    train = np.loadtxt(Config.train_file_path, skiprows=skip, delimiter=',')
    print(train.shape)
    end = time.time()
    print("Loaded training data. Time (min) = ", (end-start)/60)

    print("Loading testing data...")
    start = time.time()
    skip = 105 if Config.debug else 1
    test = np.loadtxt(Config.test_file_path, skiprows=skip, delimiter=',')
    print(test.shape)
    end = time.time()
    print("Loaded testing data. Time (min) = ", (end-start)/60)

    # Slice data into features (x) and labels (y)

    x_train = train[:,1:]  # exclude first column
    y_train = train[:,:1]  # get first column
    x_test = test[:,1:]
    y_test = test[:,:1]

    # Load feature correlation, or calculate it
    spearman_corr = calculate_correlation() if Config.recalc_corr(train) else np.load_text(Config.corr_path, delimiter = ',')

    sorted_corr = np.sort(abs(spearman_corr))[::-1] # descending

    # filter indices that are above cutoff ??
    cutoff = sorted_corr[Config.filter_size]
    spearman_index = [x for x in range(len(spearman_corr)) if abs(spearman_corr[x])<cutoff]
    spearman_complement_index = [x+1 for x in range(len(spearman_corr)) if abs(spearman_corr[x])>cutoff]
    spearman_complement_index.insert(0,0)

    #filter data
    sub_train = train[:,spearman_complement_index]
    sub_test = test[:,spearman_complement_index]
    sub_train = torch.from_numpy(sub_train).float().to(device)
    sub_test = torch.from_numpy(sub_test).float().to(device)
    x_test = sub_test[:,1:]
    y_test = sub_test[:,:1]
    
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
     
    torch.save(model.state_dict(), "model.pt")
    #output = model(x_test).data.numpy()
    #np.savetxt('CPFNN_prediction1.txt', output,delimiter = ',')
