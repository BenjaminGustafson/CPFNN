from neuralnet import NeuralNet
import torch
import pandas as pd
import numpy as np
from alepython import ale_plot

model_path = '../data/model01.pt'
model = NeuralNet(input_dim=100,hidden_dim=200,output_dim=1)
model.load_state_dict(torch.load(model_path))
model.eval()

df = pd.read_csv('../data/pandas_train100.csv')
x_train = df.iloc[:, 1:]

ale_plot(model, x_train, df.columns[1], monte_carlo=True)



###########################################
import numpy as np
import torch
import cpfnn
import pandas as pd

input = 100

model,maes,r2s = cpfnn.train_and_test('nn',input,epochs=2000, increment=100, early_stop=False)

indices = cpfnn.get_filtered_indices(input)
data = cpfnn.train[:,indices]

site_names = np.loadtxt('../data/site_names.csv', delimiter = ',', dtype=np.str)

cols = ['age']
for i in range(1, len(indices)):
    cols.append(site_names[indices[i]-1])

df = pd.DataFrame(data)
df.columns = cols

df.to_csv('../data/pandas_train10.csv', index = False)


##################################
import numpy as np
import torch
import cpfnn


input = 100

model,maes,r2s = cpfnn.train_and_test('nn',input,epochs=2000, increment=100, early_stop=False)

indices = cpfnn.get_filtered_indices(input)
x_train,y_train = cpfnn.slice_data(cpfnn.train[:,indices])



def pdp(feature_index, model, data, x_min, x_max, x_inc):
    def pdp_helper(x):
        vals = []
        for inst in data:
            newinst = inst.copy()
            newinst[feature_index] = x
            vals.append(model(torch.from_numpy(newinst).float()))
        return torch.mean(torch.Tensor(vals)).item()
    xs = np.arange(x_min, x_max, x_inc)
    return xs, list(map(pdp_helper, xs))


pdps = []
for i in range(input):
    print(i)
    xs, ys = pdp(i, model, x_train, 0, 1, 0.01) 
    pdps.append(xs)
    pdps.append(ys)

names = np.loadtxt('../data/site_names.csv', delimiter = ',', dtype=np.str)
pdp_names = []
pdp_x_data = []
pdp_y_data = y_train.T
for i in range(input):
    print(i)
    pdp_names.append(names[indices[i+1]-1])
    pdp_x_data.append(x_train[:,i])


np.savetxt('../data/nn%dpdps.csv'%input, pdps, delimiter = ',')
np.savetxt('../data/nn%dpdp_names.csv'%input, pdp_names, delimiter=',', fmt='%s')
np.savetxt('../data/nn%dpdp_x_data.csv'%input, pdp_x_data, delimiter=',')
np.savetxt('../data/nn%dpdp_y_data.csv'%input, pdp_y_data, delimiter=',')


#################################################################

import numpy as np
import matplotlib.pyplot as plt
input = 100
pdps = np.loadtxt('../data/nn%dpdps.csv'%input, delimiter = ',')
pdp_names = np.loadtxt('../data/nn%dpdp_names.csv'%input, delimiter=',', dtype=np.str)
pdp_x_data = np.loadtxt('../data/nn%dpdp_x_data.csv'%input, delimiter=',')
pdp_y_data = np.loadtxt('../data/nn%dpdp_y_data.csv'%input, delimiter=',')


def show_pdp(i):
    xs = pdps[i*2]
    ys = pdps[i*2+1]
    plt.plot(xs,ys,color = 'tab:orange')
    #plt.plot(average_xs[i], average_ys[i], color = 'tab:green')
    plt.title(pdp_names[i])
    #for x_val in pdp_x_data[i]:
    #    plt.axvline(x= x_val, color = 'k', ymax=0.05, linewidth = 0.05)
    plt.scatter(pdp_x_data[i], pdp_y_data, color = 'tab:blue')
    plt.show()

###############################################################

average_xs = []
average_ys = []
step = 0.05
for i in range(input):
    xs = []
    ys = []
    for b in np.arange(0,1,step):
        bucket = []
        for j, x in enumerate(pdp_x_data[i]):
            if b <= x < b + step:
                bucket.append(pdp_y_data[j])
        if bucket:
            xs.append(b)
            ys.append(np.mean(bucket))
    average_xs.append(xs)
    average_ys.append(ys)