import torch
import numpy as np

class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim,output_dim)
    
    def forward(self, x): 
        return self.linear(x)

def train_model(model, epochs, x_train, y_train, learningRate = 0.01):
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    for epoch in range(epochs):
        inputs = torch.from_numpy(x_train).float()
        labels = torch.from_numpy(y_train).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.item()))


def test_model(model, x_test, y_test):
    pred = model(x_test)
    abs_err = np.abs(np.subtract())
    mae = abs_err.mean()
    print('MAE {}'.format(mae))

print('Loading training data (5 min)')
train_data = np.loadtxt('/data/zhanglab/lli1/methylation/train_combat.csv', delimiter=',')
print('Loading testing data (2 min)')
test_data = np.loadtxt('/data/zhanglab/lli1/methylation/test_combat.csv', skiprows=1, delimiter=',')
corr = np.loadtxt('/data/zhanglab/bgustafs/InterpretableML/data/correlation.csv', delimiter=',')

x_train = train_data[:,1:]
y_train = train_data[:,0]
x_test = test_data[:,1:]
y_test = test_data[:,0]
weighted_train = np.matmul(x_train, corr)
weighted_test = np.matmul(x_test, corr)
#combined_train = np.stack((y_train, weighted_train), axis = 1)
#combined_test = np.stack((y_test, weighted_test), axis = 1)

model = LinearRegression(1,1)
