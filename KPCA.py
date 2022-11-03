from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np 




def rbf_kernel_pca(X, gamma, n_components):
    """ 
    KPCA with RBF
        Input Params
        Х: {NumPy ndarray}, shape= [n_examples, n_features]
        
        gamma: float
            parameter of RBF
        
        п_components: int
            number of PCs
        --------
        Returns
        alphas: {NumPy ndarray}, shape= {n_examples, k_features}
        projected data
        
        lambdas :{list}
        eigenvalues

    
    """
    # compute pairwise squared euclidean distances in n_examples x n_features
    sq_dists = pdist(X, 'sqeuclidean')

    # reshape into square matrix
    mat_sq_dists = squareform(sq_dists)

    # Compute kernel K
    K = exp(-gamma * mat_sq_dists)

    # Center K
    N = K.shape[0]

    one_n = np.ones((N,N)) / N
    K = K - one_n @ K - K @ one_n + one_n @ K  @ one_n


    # get eigenvalues and eigenvectors
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[::-1]


    # collect largest k eigenvectors
    alphas = np.column_stack([eigvecs[:,i]
                            for i in range(n_components) ])
    lambdas = [eigvecs[i] for i in range(n_components)]
    return [alphas, lambdas]                   






