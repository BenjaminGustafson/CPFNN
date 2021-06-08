import os
import pandas as pd
import torch 
import numpy as np
from scipy import stats
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# settings
class Config(object):
    #473034 303236 9607 8105 8195
    input_dim = 473034 
    hidden_dim = 100

    train_file_path = '/data/zhanglab/lli1/methylation/train_combat.csv'
    test_file_path = '/data/zhanglab/lli1/methylation/test_combat.csv'
    #train_file_path = path+'after_correlation_0.3_except_dataset_0_4datasets_train.csv'
    #test_file_path = path+'after_correlation_0.3dataset_0_4datasets_test.csv'
    use_gpu = True  # use GPU or not


# Multilayer Perceptron 
class neural_network(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim, indexes):
        super(neural_network, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

    def init_weights(self):
        init.xavier_normal(self.fc1.weight) 
        init.xavier_normal(self.fc2.weight)
        
    def forward(self, x_in, apply_softmax=False):
        
        a_1 = F.leaky_relu(self.fc1(x_in))  # activaton function added!
        y_pred = F.leaky_relu(self.fc2(a_1))
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)

        return y_pred


class Trainer(object):
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
            if ((t-1)%100==0):
                # Print the gredient
                print ("epoch: {0:02d} | loss: {1:.2f} | acc: {2:.2f}".format(t, loss, acc / len(y_train)))

    @staticmethod
    def get_accuracy(y_labels,y_pred):
        difference = torch.abs(y_labels-y_pred)
        correct = sum(list(map(lambda x: 1 if x<2 else 0,difference)))
        return correct

   # @staticmethod
   # def get_data_group()

    def test(self,x_test,y_test):
        model = self.model.eval()
        pred_test = model(x_test)
        x_arr = torch.abs(torch.sub(pred_test,y_test))
        x_arr = x_arr.cpu().data.numpy()
        x_sz = len(x_arr)

        print("MAE = " , np.sum(x_arr)/x_sz)

if torch.cuda.is_available():
    print("GPU is available to use!\n")
else:
    print("GPU is not available to use.\n")

opt = Config()


device = None
if opt.use_gpu and torch.cuda.is_available():
    device = torch.device('cuda')  
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else: 
    device = torch.device('cpu')

train = np.loadtxt(opt.train_file_path, delimiter=',')
print("Finish read training set")
test = np.loadtxt(opt.test_file_path, skiprows=1, delimiter=',')
print("Finish read test set")
spearman_corr = []
for i in range(1, train.shape[1]):
    spearman_corr.append(stats.spearmanr(train[:,0],train[:,i])[0])

spearman_corr = np.array(spearman_corr)

print('Finish read corr')

sorted_corr = np.sort(abs(spearman_corr))[::-1]
np.savetxt('sorted_cpg_correlation.csv', sorted_corr, delimiter = ',')

print(train.shape)
print(test.shape)

x_train = train[:,1:]
y_train = train[:,0].reshape(-1,1)
x_test = test[:,1:]
y_test = test[:,0].reshape(-1,1)


num_sites = [20000]#was 3000


for i in num_sites: 
    spearman_index = [x for x in range(len(spearman_corr)) if abs(spearman_corr[x])<sorted_corr[i]]
    spearman_complement_index = [x+1 for x in range(len(spearman_corr)) if abs(spearman_corr[x])>sorted_corr[i]]
    spearman_complement_index.insert(0,0)
    print(len(spearman_complement_index))
    sub_train = train[:,spearman_complement_index]
    sub_test = test[:,spearman_complement_index]
    sub_train = torch.from_numpy(sub_train).float().to(device)
    sub_test = torch.from_numpy(sub_test).float().to(device)
    x_test = sub_test[:,1:]
    y_test = sub_test[:,0].reshape(-1,1)
    
    for j in range(1):
        print('trial', j)
        start = time.time()
        model = neural_network(input_dim=i,hidden_dim=200,output_dim=1, indexes = spearman_index).to(device)


        trainer = Trainer(epoch=200,model=model,batch_size=50)
        for k in range(20):
            trainer.train_by_random(sub_train)
        end = time.time()
        torch.save(model.state_dict(), "model.pt")
        print("time elapsed (min) = ", (end-start)/60)
        trainer.test(x_test, y_test)
        #output = model(x_test).data.numpy()
        #np.savetxt('CPFNN_prediction1.txt', output,delimiter = ',')
