"""linreg.py

Contains methods for training and testing linear regression
models on the dataset from dataloader.py. Uses sklearn's 
linear_model.LinearRegression and linear_model.Ridge
classes.

Example use:
>>>import linreg
Loading training data...
Done. Time (min) = 5.025
Loading teting data...
Done. Time (min) = 1.488
>>>linreg.train_and_test(6000)
2.709085063545118
"""
import dataloader
from sklearn import linear_model
from sklearn import metrics
import pandas as pd

train = dataloader.train
test = dataloader.test

def train_and_test(num_features, model_type = 'ols', alpha = 1.0):
    """Trains a linear regression model and tests its predictions.

    Args: 
        num_features: the number of features to filter by spearman coefficient
        model_type: 'ols' - ordinary least squares
                    'ridge' - ridge regression (l2 regularization)
        alpha: parameter for ridge regression, higher alpha means more regularized
    
    Returns: 
        test_mae: the mean absolute error (MAE) of the model on the testing data.
    """
    indices = dataloader.get_filtered_indices(num_features)
    x_train = train[:,indices]
    y_train = train[:,0]
    x_test = test[:,indices]
    y_test = test[:,0]
    model = linear_model.LinearRegression()
    if model_type == 'ridge':
        model = linear_model.Ridge(alpha = alpha)
    model.fit(x_train, y_train)
    y_pred_test = model.predict(x_test)
    test_mae = metrics.mean_absolute_error(y_pred_test,y_test)
    return test_mae
    

default_ns = [2 ** n for n in range(18)]
default_alphas = [n/10 for n in range(1,11)]
def test_all(ns = default_ns, alphas = default_alphas):
    """Performs a series of tests across a range of num_features and alphas.
    
    Exports results to ../data/linreg.csv.
    """
    df = pd.DataFrame()
    df['num_features'] = ns
    test_maes = []
    for n in ns:
        print(f'training OLS with {n} features')
        test_mae = train_and_test(n, 'OLS')
        test_maes.append(test_mae)
    df['ols'] = test_maes
    for alpha in alphas:
        test_maes = []
        for n in ns:
            print(f'training ridge with {n} features and alpha = {alpha}')
            test_mae = train_and_test(n, 'ridge', alpha)
            test_maes.append(test_mae)
        df[f'Ridge (alpha={alpha})'] = test_maes
    df.to_csv('../data/linreg.csv', index=False)
    



    