'''
Utility Functions
-----------------
'''


import numpy as np
import math
from ._distributions import MultivariateNormalDistribution,TDistribution,MixedDistribution

from numba import njit



def mc_sampling(X,nsample,**kwargs):
    """Monte Carlo Sampling. 
    Generate importance sampling points and report their likelihood.

    Parameters
    ----------
    X : array-like of shape (n_train, dim_)
        List of n_train-dimensional data points.  Each row
        corresponds to a single data point.
        
    nsample : int
        Number of instances to generate. 
        
    Args:
        **method : {"bounded", "heavy_tail", "normal", "mixed"}
            Importance sampling methods to choose. Use "bounded" if all 
            entries are bounded. Use "normal" if data is concentrated. Use 
            "heavy_tail" or "mixed" if data is heavy tailed but pay attention 
            to numerical instability.
        **ruleout : float 
            Quantile for ruling out certain range of outliers. 
     
    Returns
    -------
    X_validate : array-like of shape (nsample, dim_)
        List of nsample-dimensional data points.  Each row
        corresponds to a single data point.
        
    pdf_X_validate : array-like of shape (nsample, )
        Pdf of X_validate.
    """
    dim=X.shape[1]
    if kwargs["method"] == "bounded":
        lower = np.array([np.quantile(X[:,i],kwargs["ruleout"]) for i in range(dim)])
        upper = np.array([np.quantile(X[:,i],1-kwargs["ruleout"]) for i in range(dim)])
        np.random.seed(kwargs["seed"])
        return np.random.rand(int(nsample),dim)*(upper-lower)+lower,np.ones(int(nsample))/np.prod(upper-lower)
    if kwargs["method"] == "heavy_tail":
        density = TDistribution(loc = np.zeros(dim),scale = np.ones(dim),df = 2/3)
        np.random.seed(kwargs["seed"])
        return density.generate(int(nsample))
    if kwargs["method"] == "normal":
        density = MultivariateNormalDistribution(mean = X.mean(axis = 0),cov = np.diag(np.diag(np.cov(X.T))))
        np.random.seed(kwargs["seed"])
        return density.generate(int(nsample))
    if kwargs["method"] == "mixed":
        density1 = MultivariateNormalDistribution(mean = X.mean(axis = 0),cov = np.diag(np.diag(np.cov(X.T))))
        density2 = TDistribution(loc=np.zeros(dim),scale = np.ones(dim),df = 2/3)
        density_seq = [density1, density2]
        prob_seq = [0.7,0.3]
        densitymix = MixedDistribution(density_seq, prob_seq)
        return densitymix.generate(int(nsample))

@njit
def weight_selection(beta,cut_off):
    """Find the optimization solution of optimal weights. 

    Parameters
    ----------
    Beta : array-like of shape (potentialNeighbors, )
        Array of rescaled distance vector. Suppose to be increasing.
        
    cut_off : int
        Number of neighbors for cutting AWNN to KNN. 

    Returns
    -------
    estAlpha: array-like of shape (potentialNeighbors, )
        Solved weights. 
        
    alphaIndexMax: int
        Solved number of neighbors.
        
    Reference
    ---------
    Oren Anava and Kfir Levy. k*-nearest neighbors: From global to local. 
    Advances in neural information processing systems, 29, 2016.
    """
    potentialNeighbors = len(beta)
    alphaIndexMax = 0
    lamda = beta[0]+1 
    Sum_beta = 0
    Sum_beta_square = 0
    # iterates for k
    while ( lamda>beta[alphaIndexMax] ) and (alphaIndexMax<potentialNeighbors):
        # update max index
        alphaIndexMax +=1
        # updata sum beta and sum beta square
        Sum_beta += beta[alphaIndexMax-1]
        Sum_beta_square += (beta[alphaIndexMax-1])**2
        # calculate lambda
        if  alphaIndexMax  + (Sum_beta**2 - alphaIndexMax * Sum_beta_square)>=0:
            lamda = (1/alphaIndexMax) * ( Sum_beta + math.sqrt( alphaIndexMax  + (Sum_beta**2 - alphaIndexMax * Sum_beta_square) ) )
        else:
            alphaIndexMax -= 1
            break
    # estimation
    estAlpha = np.zeros(potentialNeighbors)
    if alphaIndexMax<cut_off:
        estAlpha[cut_off-1] = 1
        return estAlpha,cut_off
    for j in range(alphaIndexMax):
        estAlpha[j] = lamda-beta[j]
    estAlpha = estAlpha/np.linalg.norm(estAlpha,ord=1)
    return estAlpha,alphaIndexMax

def knn(X,tree,k,n,dim,vol_unitball):
    """Standard k-NN density estimation. 

    Parameters
    ----------
    X : array-like of shape (n_test, dim_)
        List of n_test-dimensional data points.  Each row
        corresponds to a single data point.
        
    tree_ : "KDTree" instance
        The tree algorithm for fast generalized N-point problems.
        
    k : int
        Number of neighbors to consider in estimaton. 
        
    n : int
        Number of traning samples.
        
    dim : int
        Number of features.
        
    vol_unitball : float
        Volume of dim_ dimensional unit ball.

    Returns
    -------
    log_density: array-like of shape (n_test, ).
        Estimated log-density of test samples.
        
    Reference
    ---------
    Sanjoy Dasgupta and Samory Kpotufe. Optimal rates for k-nn density and mode 
    estimation. Advances in Neural Information Processing Systems, 27, 2014.
    """
    if len(X.shape) == 1:
        X = X.reshape(1,-1).copy()
    distance_matrix,_ = tree.query(X,k+1)
    # rule out self testing
    if (distance_matrix[:,0]==0).all():
        log_density = np.log(k/n/vol_unitball/(distance_matrix[:,k]**dim))
    else:
        log_density = np.log(k/n/vol_unitball/(distance_matrix[:,k-1]**dim))
    return log_density

    # TODO: add alpha indexed choice of weights
def wknn(X,tree,k,n,dim,vol_unitball):
    """Weighted k-NN density estimation. 

    Parameters
    ----------
    X : array-like of shape (n_test, dim_)
        List of n_test-dimensional data points.  Each row
        corresponds to a single data point.
        
    tree_ : "KDTree" instance
        The tree algorithm for fast generalized N-point problems.
        
    k : int
        Number of neighbors to consider in estimaton. 
        
    n : int
        Number of traning samples.
        
    dim : int
        Number of features.
        
    vol_unitball : float
        Volume of dim_ dimensional unit ball.

    Returns
    -------
    log_density: array-like of shape (n_test, ).
        Estimated log-density of test samples.
        
    Reference
    ---------
    Gérard Biau, Frédéric Chazal, David Cohen-Steiner, Luc Devroye, and Carlos 
    Rodríguez. A weighted k-nearest neighbor density estimate for geometric 
    inference. Electronic Journal of Statistics, 5(none):204 – 237, 2011. 
    doi: 10.1214/11-EJS606. URL https://doi.org/ 10.1214/11-EJS606.
    """
    if len(X.shape) == 1:
        X = X.reshape(1,-1).copy()
    distance_matrix,_ = tree.query(X,k+1)
    # rule out self testing
    if (distance_matrix[:,0] == 0).all():
        log_density = np.log((k+1)*k/2/n/vol_unitball/(distance_matrix[:,1:]**dim).sum(axis=1))
    else:
        log_density = np.log((k+1)*k/2/n/vol_unitball/(distance_matrix[:,:-1]**dim).sum(axis=1))
    return log_density

def tknn(X,tree,k,n,dim,vol_unitball,threshold_num,threshold_r):
    """Adaptive k-NN density estimation. 

    Parameters
    ----------
    X : array-like of shape (n_test, dim_)
        List of n_test-dimensional data points.  Each row
        corresponds to a single data point.
        
    tree_ : "KDTree" instance
        The tree algorithm for fast generalized N-point problems.
        
    k : int
        Number of neighbors to consider in estimaton. 
        
    n : int
        Number of traning samples.
        
    dim : int
        Number of features.
        
    vol_unitball : float
        Volume of dim_ dimensional unit ball.
        
    threshold_r : float
        Threshold paramerter in AKNN to identify tail instances. 
    threshold_num : int 
        Threshold paramerter in AKNN to identify tail instances. 
     

    Returns
    -------
    log_density: array-like of shape (n_test, ).
        Estimated log-density of test samples.
        
    Reference
    ---------
    Puning Zhao and Lifeng Lai. Analysis of knn density estimation, 2020.
    """
    if len(X.shape) == 1:
        X = X.reshape(1,-1).copy()
    distance_matrix,_ = tree.query(X,k+1)
    # identify tail instances
    mask = tree.query_radius(X, r = threshold_r,
                           count_only = True)>threshold_num
    masked_estimation = np.array([i/n/vol_unitball/threshold_r**dim 
                                  for i in tree.query_radius(X, r=threshold_r, count_only = True)]) *np.logical_not(mask)
    # rule out self testing
    if (distance_matrix[:,0] == 0).all():
        log_density = np.log(k/n/vol_unitball/(distance_matrix[:,k]**dim)*mask+1e-30)+masked_estimation
    else:
        log_density = np.log(k/n/vol_unitball/(distance_matrix[:,k-1]**dim)*mask+1e-30)+masked_estimation
    return log_density

def bknn(X,tree,n,dim,vol_unitball,kmax,C,C2):
    """Balanced k-NN density estimation. 

    Parameters
    ----------
    X : array-like of shape (n_test, dim_)
        List of n_test-dimensional data points.  Each row
        corresponds to a single data point.
        
    tree_ : "AdaptiveKDTree" instance
        The tree algorithm for fast generalized N-point problems.
        
    kmax : int
        Number of maximum neighbors to consider in estimaton. 
        
    n : int
        Number of traning samples.
        
    dim : int
        Number of features.
        
    vol_unitball : float
        Volume of dim_ dimensional unit ball.
        
    C : float 
        Scaling paramerter in BKNN.
        
    C2 : float 
        Threshold paramerter in BKNN.
     
    Returns
    -------
    log_density: array-like of shape (n_test, ).
        Estimated log-density of test samples.
        
    Reference
    ---------
    Julio A Kovacs, Cailee Helmick, and Willy Wriggers. A balanced approach 
    to adaptive probability density estimation. Frontiers in molecular 
    biosciences, 4:25, 2017.
    """
    if len(X.shape) == 1:
        X = X.reshape(1,-1).copy()
    log_density=[]
    distance_vec, k_vec = tree.adaptive_query(X,beta = dim,C = C2,max_neighbor = kmax)
    log_density.append(np.log(k_vec*C/n/vol_unitball/(distance_vec**dim)+1e-30))
    return np.array(log_density)

# TODO: add comments
def aknn(X,tree,n,dim,vol_unitball,kmax,C):
    """Balanced k-NN density estimation. 

    Parameters
    ----------
    X : array-like of shape (n_test, dim_)
        List of n_test-dimensional data points.  Each row
        corresponds to a single data point.
        
    tree_ : "AdaptiveKDTree" instance
        The tree algorithm for fast generalized N-point problems.
        
    kmax : int
        Number of maximum neighbors to consider in estimaton. 
        
    n : int
        Number of traning samples.
        
    dim : int
        Number of features.
        
    vol_unitball : float
        Volume of dim_ dimensional unit ball.
        
    C : float 
        Threshold paramerter in BKNN.
     

    Returns
    -------
    log_density: array-like of shape (n_test, ).
        Estimated log-density of test samples.
        

    """
    if len(X.shape) == 1:
        X = X.reshape(1,-1).copy()
    log_density = []
    distance_vec, k_vec = tree.adaptive_query(X,beta = dim,C = C,max_neighbor = kmax)
    log_density.append(np.log(k_vec/n/vol_unitball/(distance_vec**dim)+1e-30))
    return np.array(log_density)
    
    