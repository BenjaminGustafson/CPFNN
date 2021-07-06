import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans

data = np.loadtxt('../data/top1ktrain.csv',delimiter=',')

def plot_site(n):
    x = data[:,n]
    y = data[:,0]
    plt.xlim(0,1)
    plt.ylim(0,100)
    plt.scatter(x,y)
    z = np.polyfit(x, y, 1)
    y_hat = np.poly1d(z)(x)
    plt.plot(x, y_hat, "r--", lw=1)
    text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(y,y_hat):0.3f}$"
    plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
        fontsize=14, verticalalignment='top')
    plt.show()

def kmeans(n):
    x = data[:,n]
    y = data[:,0]
    plt.show()

