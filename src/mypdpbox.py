import matplotlib.pyplot as plt
from pdpbox import pdp
from neuralnet import NeuralNet
import torch
import pandas as pd

input = 100
model_path = '../data/nn{}.pt'.format(input)
model = NeuralNet(input_dim=input,hidden_dim=200,output_dim=1)
model.load_state_dict(torch.load(model_path))
model.eval()

train_df = pd.read_csv('../data/pandas_train{}.csv'.format(input))

features = train_df.columns[1:]

def show_pdp(n):
    my_pdp = pdp.pdp_isolate(
        model=model, dataset=train_df, model_features=features, feature=features[n]
    )
    fig, axes = pdp.pdp_plot(my_pdp, features[n], plot_lines=True, frac_to_plot=100)
    plt.show(fig)
