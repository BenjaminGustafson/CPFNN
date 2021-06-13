import sys; sys.path.append('..')
import torch
import CPFNN.cpfnn as cpfnn
import lime
import lime.lime_tabular
from lime import submodular_pick

input = 20
hidden = 100

if __name__ == "__main__":
    
    device = torch.device('cpu')
    
    train = cpfnn.load_training_data()
    test = cpfnn.load_testing_data()
    indices = cpfnn.get_filtered_indices(filter_size = input, recalc_corr = False)
    sub_train = train[:,indices]
    sub_test = test[:,indices]
    x_train, y_train = cpfnn.slice_data(sub_train)
    x_test, y_test = cpfnn.slice_data(sub_test)
    sub_train_tensor = torch.from_numpy(sub_train).float().to(device)
    sub_test_tensor = torch.from_numpy(sub_test).float().to(device)
    x_train_tensor, y_train_tensor = cpfnn.slice_data(sub_train_tensor)
    x_test_tensor, y_test_tensor = cpfnn.slice_data(sub_test_tensor)

    model_path = '../../data/model06/model06.pt'
    model = cpfnn.NeuralNet(input_dim=input,hidden_dim=hidden,output_dim=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    def predict(x_in):
        """A wrapper function for the model's prediction function. Converts 
           numpy array to torch's Tensor.
        """
        x_in_tensor = torch.from_numpy(x_in).float().to(device)
        return model.forward(x_in_tensor)


    explainer = lime.lime_tabular.LimeTabularExplainer(x_train, mode='regression')
    
    #exp = explainer.explain_instance(x_test[0], predict, num_features=20)
    #exp.save_to_file('../../data/model06exp0.html')
    sp_obj = submodular_pick.SubmodularPick(explainer, x_train, predict, num_exps_desired= 5, num_features = 20)
    for i,exp in enumerate(sp_obj.sp_explanations):
        exp.save_to_file('../../data/model06/exp%d' % i)
