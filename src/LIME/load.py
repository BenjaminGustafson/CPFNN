import sys; sys.path.append('..')
import torch
import CPFNN.prefilter as prefilter

model_path = '../../data/model.pt'
config = prefilter.Config
model = prefilter.NeuralNet(input_dim=config.filter_size,hidden_dim=config.hidden_dim,output_dim=config.output_dim)
model.load_state_dict(torch.load(model_path))
model.eval()


