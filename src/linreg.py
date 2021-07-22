import dataloader
from sklearn import linear_model
from sklearn import metrics
import pandas as pd


train = dataloader.train
test = dataloader.test

def train_and_test(num_features, model_type = 'ols', alpha = 1.0):
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
    
def test_all():
    ns = [2 ** n for n in range(18)]  # or [1.5 ** n for n in range(1,30)]
    alphas = [10 ** p for p in range(-3,2)]
    df = pd.DataFrame()
    df['num_features'] = ns
    test_maes = []
    for n in ns:
        print('training OLS')
        test_mae = train_and_test(n, 'OLS')
        test_maes.append(test_mae)
    df['ols'] = test_maes
    for alpha in alphas:
        test_maes = []
        for n in ns:
            test_mae = train_and_test(n, 'ridge')
            test_maes.append(test_mae)
        df[f'Ridge (alpha={alpha})'] = test_maes
    df.to_csv('../data/linreg.csv', index=False)
    



    