import torch
import matplotlib.pyplot as plt
import math

class LinearRegression:
    def fit(self, X, y, method, learning_rate=0.01, iterations=500, batch_size=32, k = 0.01):
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
        X = torch.cat([(X), torch.ones_like(y)], dim=1)
        rows, cols = X.size()
        if method == 'solve':
            """
            weights = (X^T X)^-1 X^T y
            """
            if rows >= cols == torch.matrix_rank(X):
                self.weights = torch.matmul(
                    torch.matmul(
                        torch.inverse(
                            torch.matmul(
                                torch.transpose(X, 0, 1),
                                X)),
                        torch.transpose(X, 0, 1)),
                    y)
            else:
                print('X has not full column rank. method=\'solve\' cannot be used.')
        if method == 'ridge':
            """
            weights = (X^T X + k I_d)^-1 X^T y
            """
            if rows >= cols == torch.matrix_rank(X):
                self.weights = torch.matmul(
                    torch.matmul(
                        torch.inverse(
                            torch.add(
                                torch.matmul(
                                    torch.transpose(X, 0, 1),
                                    X),
                                torch.mul(
                                    torch.eye(cols,cols),
                                    k))),
                        torch.transpose(X, 0, 1)),
                    y)
            #line too long^
            else:
                print('X has not full column rank. method=\'ridge\' cannot be used.')
        elif method == 'sgd':
            self.weights = torch.normal(mean=0, std=1/cols, size=(cols, 1), dtype=torch.float64)
            for i in range(iterations):
                Xy = torch.cat([X, y], dim=1)
                Xy = Xy[torch.randperm(Xy.size()[0])]
                X, y = torch.split(Xy, [Xy.size()[1]-1, 1], dim=1)
                for j in range(int(math.ceil(rows/batch_size))):
                    start, end = batch_size*j, min(batch_size*(j+1), rows)
                    Xb = torch.index_select(X, 0, torch.arange(start, end))
                    yb = torch.index_select(y, 0, torch.arange(start, end))
                    
                    self.weights.requires_grad_(True)
                    diff = torch.matmul(Xb, self.weights) - yb
                    loss = torch.matmul(torch.transpose(diff, 0, 1), diff)
                    loss.backward()
                    
                    self.weights = (self.weights - learning_rate*self.weights.grad).detach()
        else:
            print(f'Unknown method: \'{method}\'')
        
        return self
    
    def predict(self, X):
        X = torch.from_numpy(X).float()
        if not hasattr(self, 'weights'):
            print('Cannot predict. You should call the .fit() method first.')
            return
        
        X = torch.cat([X, torch.ones((X.size()[0], 1))], dim=1)
        
        if X.size()[1] != self.weights.size()[0]:
            print(f'Shapes do not match. {X.size()[1]} != {self.weights.size()[0]}')
            return
        
        return torch.matmul(X, self.weights)
    
    def rmse(self, X, y):
        y = torch.from_numpy(y).float()
        y_hat = self.predict(X)
        
        if y_hat is None:
            return
        
        return torch.sqrt(torch.mean(torch.square(y_hat - y)))
