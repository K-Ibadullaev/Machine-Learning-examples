from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
def plot_decision_regions(X, y, classifier, test_idx=None,
                          resolution = 0.02):
    markers = ('s', 'x', 'o', '4', '4')
    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # show the decision region
    x1_min, x1_max = X[:, 0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # display by class
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[ y == cl, 0],
                    y=X[y == cl, 1],
                   alpha=0.8,
                   c=colors[idx],
                   marker=markers[idx],
                   label =cl)
        
    if test_idx:
        X_ts, y_ts = X[test_idx,:], y[test_idx]
        plt.scatter( X_ts[:,0],  X_ts[:,1],c='w',
                   edgecolor='k', alpha=1.0,linewidth=1, marker='o',
                   s=100, label='Test set')