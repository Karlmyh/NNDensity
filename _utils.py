'''
Utility Functions
-----------------
'''

import numpy as np
import math
from scipy.stats import multivariate_normal
from scipy.stats import t

from numba import njit


'''
Distribution Objects
-------------------- 
For sample generation and likelihood evaluation w.r.t. certain distribution. 

'''
class Distribution(object): 
    def __init__(self):
        pass
    def generate(self, num_samples):
        sample_X = self.sampling(num_samples) 
        pdf_true = self.density(sample_X) 
        return sample_X, pdf_true
    def sampling(self, num_samples): 
        pass
    def density(self, sample_X): 
        pass
    
class MultivariateNormalDistribution(Distribution): 
    def __init__(self, mean, cov):
        super(MultivariateNormalDistribution, self).__init__()
        self.mean = mean
        self.cov = cov
        self.dim=np.array([mean]).ravel().shape[0]
        
    def sampling(self, num_samples):
            
        return multivariate_normal.rvs(mean=self.mean,
                                       cov=self.cov,
                                       size=num_samples).reshape(-1,self.dim)
    def density(self, sample_X):
        return multivariate_normal.pdf(sample_X, mean=self.mean, cov=self.cov)
    
    
class TDistribution(Distribution):
    def __init__(self,scale,loc,df):
        super(TDistribution, self).__init__()
        self.scale = np.array(scale).ravel()
        self.loc = np.array(loc).ravel()
        self.df=df
  
        self.dim=len(np.array(scale).ravel())
    def sampling(self, num_samples):
        
        sample_X=[]
        for i in range(self.dim):
            sample_Xi=t.rvs(loc=self.loc[i],scale=self.scale[i],size=num_samples,df=self.df)
            
            
            sample_X.append(sample_Xi.reshape(-1,1))
        sample_X = np.concatenate(sample_X, axis=1)
        return sample_X
    def density(self, sample_X):
        return np.prod(t.pdf(sample_X,loc=self.loc,scale=self.scale,df=self.df), axis=1)
 
class MixedDistribution(Distribution):
    def __init__(self, density_seq, prob_seq):
        super(MixedDistribution, self).__init__()
        self.density_seq = density_seq
        self.prob_seq = prob_seq
        self.num_mix = len(density_seq)
        self.dim=self.density_seq[0].dim
        
    def sampling(self, num_samples):
        rd_idx = np.random.choice(self.num_mix,size=num_samples, 
                                  replace=True, p=self.prob_seq)
        sample_X = []
        for i in range(self.num_mix):
            num_i = np.sum(rd_idx == i)
            density = self.density_seq[i]
            sample_Xi, _ = density.generate(num_i)
            sample_X.append(sample_Xi)
        sample_X = np.concatenate(sample_X, axis=0)
        np.random.shuffle(sample_X)
        return sample_X
    
    # return density at given vector of point
    def density(self, sample_X):
        num_samples = sample_X.shape[0]
        pdf_true = np.zeros(num_samples, dtype=np.float64) 
        for i in range(self.num_mix):
            prob = self.prob_seq[i]
            density = self.density_seq[i]
            pdf_true += prob * density.density(sample_X)
        return pdf_true
    
    




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
    
    if kwargs["method"]=="bounded":
        lower=np.array([np.quantile(X[:,i],kwargs["ruleout"]) for i in range(dim)])
        upper=np.array([np.quantile(X[:,i],1-kwargs["ruleout"]) for i in range(dim)])
            
        np.random.seed(kwargs["seed"])
        return np.random.rand(int(nsample),dim)*(upper-lower)+lower,np.ones(int(nsample))/np.prod(upper-lower)
    if kwargs["method"]=="heavy_tail":
        density=TDistribution(loc=np.zeros(dim),scale=np.ones(dim),df=2/3)
        np.random.seed(kwargs["seed"])
        return density.generate(int(nsample))
    if kwargs["method"]=="normal":
        density=MultivariateNormalDistribution(mean=X.mean(axis=0),cov=np.diag(np.diag(np.cov(X.T))))
        np.random.seed(kwargs["seed"])
        return density.generate(int(nsample))
    
    if kwargs["method"]=="mixed":
        density1 = MultivariateNormalDistribution(mean=X.mean(axis=0),cov=np.diag(np.diag(np.cov(X.T)))) 
        density2 = TDistribution(loc=np.zeros(dim),scale=np.ones(dim),df=2/3)
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
    
    potentialNeighbors=len(beta)
    alphaIndexMax=0
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
            alphaIndexMax-=1
            break
    
    # estimation
    estAlpha=np.zeros(potentialNeighbors)

    
    if alphaIndexMax<cut_off:
        estAlpha[cut_off-1]=5
        return estAlpha,cut_off
    
    
    for j in range(alphaIndexMax):
        estAlpha[j]=lamda-beta[j]
    
    
    estAlpha=estAlpha/np.linalg.norm(estAlpha,ord=1)
    
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
    if len(X.shape)==1:
        X=X.reshape(1,-1).copy()
        
    distance_matrix,_=tree.query(X,k+1)
    
    # rule out self testing
    if (distance_matrix[:,0]==0).all():
        log_density= np.log(k/n/vol_unitball/(distance_matrix[:,k]**dim)) 
    else:
        log_density= np.log(k/n/vol_unitball/(distance_matrix[:,k-1]**dim))

    return log_density
    
    
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
    if len(X.shape)==1:
        X=X.reshape(1,-1).copy()

    distance_matrix,_=tree.query(X,k+1)
    
    # rule out self testing
    if (distance_matrix[:,0]==0).all():
        log_density= np.log((k+1)*k/2/n/vol_unitball/(distance_matrix[:,1:]**dim).sum(axis=1))
    else:
        log_density= np.log((k+1)*k/2/n/vol_unitball/(distance_matrix[:,:-1]**dim).sum(axis=1))
    
    return log_density
    
    
def aknn(X,tree,k,n,dim,vol_unitball,**kwargs):
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
        
    Args:
        **threshold_r : float
            Threshold paramerter in AKNN to identify tail instances. 
        **threshold_num : int 
            Threshold paramerter in AKNN to identify tail instances. 
     

    Returns
    -------
    log_density: array-like of shape (n_test, ).
        Estimated log-density of test samples.
        
    Reference
    ---------
    Puning Zhao and Lifeng Lai. Analysis of knn density estimation, 2020.
    """

    if len(X.shape)==1:
        X=X.reshape(1,-1).copy()

    distance_matrix,_=tree.query(X,k+1)
    
    # identify tail instances
    mask=tree.query_radius(X, r=kwargs["threshold_r"], 
                           count_only=True)<kwargs["threshold_num"]
    
    # rule out self testing
    if (distance_matrix[:,0]==0).all():
        log_density=np.log(k/n/vol_unitball/(distance_matrix[:,k]**dim)*mask+1e-30)
    else:
        log_density=np.log(k/n/vol_unitball/(distance_matrix[:,k-1]**dim)*mask+1e-30)
        
    return log_density
    
    