import sys; sys.path.append('..')
import torch
import CPFNN.cpfnn as cpfnn

input = 3000
hidden = 200
output = 1

train = cpfnn.load_training_data()
filtered_indices = cpfnn.get_filtered_indices(filter_size = input, recalc_corr = False)
sub_train = cpfnn.filter_data(train, filtered_indices)
# x = features, y = labels
x_train, y_train = cpfnn.slice_data(sub_train)

model_path = '../../data/model01.pt'
model = cpfnn.NeuralNet(input_dim=3000,hidden_dim=200,output_dim=1)
model.load_state_dict(torch.load(model_path))
model.eval()


