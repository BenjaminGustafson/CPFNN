"""
explain.py

Runs SP-LIME on a model from cpfnn.py. Saves explanations as html files.  
"""
import torch
import cpfnn
import sys; sys.path.append('../../lime')
import lime
from lime import lime_tabular
from lime import submodular_pick


#Train model
input = 10
model, mae, r2 = cpfnn.train_and_test('nn',input)
indices = cpfnn.get_filtered_indices(input)
x_train, y_train = cpfnn.slice_data(cpfnn.train[:,indices])

#Make explainer
def predict(x):
    """A wrapper function for the model's prediction function. Converts 
        numpy array to torch's Tensor.
    """
    return model(torch.from_numpy(x).float())

explainer = lime_tabular.LimeTabularExplainer(x_train, mode='regression')

#SP-LIME
sp_obj = submodular_pick.SubmodularPick(explainer, x_train, predict, num_exps_desired= 5, num_features = 20)
for i,exp in enumerate(sp_obj.sp_explanations):
    exp.save_to_file('../data/nn{}exp{}'.format(input,i))

#Get individual explanation
#exp = explainer.explain_instance(x_test[0], predict, num_features=20)
#exp.save_to_file('../../data/model06exp0.html')