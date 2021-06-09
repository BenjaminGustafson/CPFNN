import sys; sys.path.append('..')
import torch
import CPFNN.cpfnn as cpfnn

model_path = '../../data/model01.pt'
model = cpfnn.NeuralNet(input_dim=3000,hidden_dim=200,output_dim=1)
model.load_state_dict(torch.load(model_path))
model.eval()


