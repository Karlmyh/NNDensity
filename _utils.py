import numpy as np

__all__ = ['grid_sampling']

def grid_sampling(X,nsample=0):
    
    dim=X.shape[1]
    
    potentialNeighbors=X.shape[0]
    if nsample==0:
        nsample=2*potentialNeighbors*dim**(dim*0.5)
    lower=np.array([np.min(X[:,i]) for i in range(dim)])
    upper=np.array([np.max(X[:,i]) for i in range(dim)])
        
    
    return np.random.rand(int(nsample),dim)*(upper-lower)+lower