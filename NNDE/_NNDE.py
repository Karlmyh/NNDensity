import numpy as np
import math
from sklearn.neighbors import KDTree

from ._utils import mc_sampling,weight_selection,knn,wknn,tknn,bknn




class NNDE(object):
    def __init__(
        self,
        metric="euclidean",
        leaf_size=40,
        seed=1,
        score_criterion="MISE",
        sampling_stratigy="bounded",
        max_neighbors="auto",
        score_validate_scale="auto"
    ):
        self.metric = metric
        self.leaf_size=leaf_size
        self.seed=seed
        self.score_criterion=score_criterion
        self.sampling_stratigy=sampling_stratigy
        self.max_neighbors=max_neighbors
        self.score_validate_scale=score_validate_scale
        
        
        if metric not in KDTree.valid_metrics:
            raise ValueError("invalid metric: '{0}'".format(metric))
        
        self.log_density=None
        
        
        

    def fit(self, X, y=None):
       
        if self.max_neighbors=="auto":
            self.max_neighbors_=min(int(X.shape[0]*(2/3)),10000)
        
            
        if self.score_validate_scale=="auto":
            self.score_validate_scale_=self.n_train_*(self.dim_*2)
        

        self.tree_ = KDTree(
            X,
            metric=self.metric,
            leaf_size=self.leaf_size,
        )
        
        self.dim_=X.shape[1]
        self.n_train_=X.shape[0]
        self.vol_unitball_=math.pi**(self.dim_/2)/math.gamma(self.dim_/2+1)
        
        
        return self
    
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in ['k','threshold_r','threshold_num','C']:
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out
    
    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)


        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            setattr(self, key, value)
            valid_params[key] = value

        return self
    
    def score_samples(self, X):
        pass
    
    
    def predict(self,X,y=None):

        
        self.log_density=self.score_samples(X)
        
        return self.log_density
    
    
    def compute_KL(self,X):

        
        # Monte Carlo estimation of integral
        kwargs={"ruleout":0.01,"method":self.sampling_stratigy,"seed":self.seed}
        X_validate,pdf_X_validate=mc_sampling(X,nsample=self.score_validate_scale_,**kwargs)
        validate_log_density=self.score_samples(X_validate)
        
        # if density has been computed, then do not update object attribute
        if self.log_density is None:
            self.log_density=self.score_samples(X)
        
        return self.log_density.mean()-(np.exp(validate_log_density)/pdf_X_validate).mean()
    
        
    def compute_MISE(self,X):

        
        # Monte Carlo estimation of integral
        kwargs={"ruleout":0.01,"method":self.sampling_stratigy,"seed":self.seed}
        X_validate,pdf_X_validate=mc_sampling(X,nsample=self.score_validate_scale_,**kwargs)
        validate_log_density=self.score_samples(X_validate)
        
        # if density has been computed, then do not update object attribute
        if self.log_density is None:
            self.log_density=self.score_samples(X)
        
        return 2*np.exp(self.log_density).mean()-(np.exp(2*validate_log_density)/pdf_X_validate).mean()
    
    def compute_ANLL(self,X):

        
        # if density has been computed, then do not update object attribute
        if self.log_density is None:
            self.log_density=self.score_samples(X)

        return -self.log_density.mean()

    def score(self, X, y=None):

        self.ANLL=self.compute_ANLL(X)
        
        if self.score_criterion=="KL":
            self.KL=self.compute_KL(X)
            return self.KL
        elif self.score_criterion=="MISE":
            self.MISE=self.compute_MISE(X)
            return self.MISE
    
    
    
class AWNN(NNDE):
    def __init__(self,
                 C=1,
                 cut_off=5,
                 save_weights=False,
                 threshold_num=5,
                 threshold_r=0.5,
                 k=2):
        super(AWNN, self).__init__()
        self.C=C
        self.cut_off=cut_off
        self.save_weights=save_weights
        
        
    def score_samples(self, X):
    

        """Compute the log-likelihood of each sample under the model.

        Parameters
        ----------
        X : array-like of shape (n_test, dim_)
            An array of points to query.  Last dimension should match dimension
            of training data (dim_).

        Returns
        -------
        log_density : ndarray of shape (n_test,)
            Log-likelihood of each sample in `X`.
        
        """
        
        n_test=X.shape[0]
        
        log_density=np.zeros(n_test)
        if self.save_weights:
            self.alpha=np.zeros((n_test,self.n_train_-1))
        
        for i in range(n_test):
            
            distance_vec,_=self.tree_.query(X[i].reshape(1,-1),self.max_neighbors_+1)
            distance_vec=distance_vec[0]
            # rule out self testing
            if distance_vec[0]==0:
                distance_vec=distance_vec[1:]
            else:
                distance_vec=distance_vec[:-1]
                
            beta=self.C*distance_vec
            
            
            estAlpha,alphaIndexMax=weight_selection(beta,cut_off=self.cut_off)
            # rule out self testing
            
           
            
            # query more points if all points are used
            if alphaIndexMax==self.max_neighbors_:
                
                distance_vec,_=self.tree_.query(X[i].reshape(1,-1),self.n_train_)
                distance_vec=distance_vec[0]
                # rule out self testing
                if distance_vec[0]==0:
                    distance_vec=distance_vec[1:]
                    beta=self.C*distance_vec
                    estAlpha,alphaIndexMax=weight_selection(beta,cut_off=self.cut_off)
                    
                    
                else:
                    distance_vec=distance_vec[:-1]
                    beta=self.C*distance_vec
                    estAlpha,alphaIndexMax=weight_selection(beta,cut_off=self.cut_off)
            if self.save_weights:        
                self.alpha[i,:estAlpha.shape[0]]=estAlpha
                    
            density_num=np.array([k for k in range(1,alphaIndexMax+1)]).dot(estAlpha[:alphaIndexMax])
            density_den=np.array([distance_vec[:alphaIndexMax]**self.dim_]).dot(estAlpha[:alphaIndexMax])
            
            if density_num<=0 or density_den<=0:
                log_density[i]=-30
            else:
                log_density[i] = math.log(density_num)-math.log(density_den)
                
        log_density-=np.log(self.n_train_*self.vol_unitball_)
   
        return log_density
    
    
class KNN(NNDE):
    def __init__(self,
                 C=1,
                 cut_off=5,
                 save_weights=False,
                 threshold_num=5,
                 threshold_r=0.5,
                 k=2):
        super(KNN, self).__init__()
        self.k=k
        
        
    def score_samples(self, X):
    

        log_density=knn(X,self.tree_,self.k,self.n_train_,self.dim_,
                         self.vol_unitball_)

        return log_density
    
class WKNN(NNDE):
    def __init__(self,
                 C=1,
                 cut_off=5,
                 save_weights=False,
                 threshold_num=5,
                 threshold_r=0.5,
                 k=2):
        super(KNN, self).__init__()
        self.k=k
        
        
    def score_samples(self, X):
    

        log_density=wknn(X,self.tree_,self.k,self.n_train_,self.dim_,
                         self.vol_unitball_)

        return log_density
    
    
class TKNN(NNDE):
    def __init__(self,
                 C=1,
                 cut_off=5,
                 save_weights=False,
                 threshold_num=5,
                 threshold_r=0.5,
                 k=2):
        super(TKNN, self).__init__()
        self.threshold_num=threshold_num
        self.threshold_r=threshold_r
        
        
    def score_samples(self, X):
    

        log_density=tknn(X,self.tree_,self.k,self.n_train_,self.dim_,
                        self.vol_unitball_,self.threshold_num,self.threshold_r)

        return log_density
    
    
    
class BKNN(NNDE):
    def __init__(self,
                 C=0.5,
                 cut_off=5,
                 save_weights=False,
                 threshold_num=5,
                 threshold_r=0.5,
                 k=2):
        super(BKNN, self).__init__()
        self.C=C

    def score_samples(self, X):
        
        if self.dim_==1:
            self.C2=np.sqrt(np.cov(X.T))
        else:
            self.C2=np.sqrt(np.linalg.det(np.cov(X.T)))
        self.C2=self.C2*(0.028*self.n_train_**(0.8))**(1/self.dim_)
        self.kmax=int(self.n_train_**0.5)
        
        log_density=bknn(X,self.tree_,self.k,self.n_train_,self.dim_,
                        self.vol_unitball_,self.kmax,self.C,self.C2)

        return log_density






    