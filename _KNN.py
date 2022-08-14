"""
Nearest Neighbor Based Density Estimation
-----------------------------------------------------
"""

import numpy as np
import math

from ._utils import mc_sampling
from sklearn.neighbors import KDTree
from ._NNAlgorithms import knn,wknn,aknn,bknn





class KNN(object):
    """Nearest Neighbor Based Density Estimation

    Incorporate k-NN density estimation (KNN), weighted k-NN density estimaton 
    (WKNN) and adaptive k-NN density estimation (AKNN). See reference. 

    Parameters
    ----------
    k : int, default=2
        Number of neighbors to consider in estimaton. 
        
    threshold_r : float, default= 0.5
        Threshold paramerter in AKNN to identify tail instances. 
        
    threshold_num: int, default= 5
        Threshold paramerter in AKNN to identify tail instances. 
    
    C: float, default= 1
        Scaling paramerter in BKNN.
        
    metric : str, default='euclidean'.
        The distance metric to use.  Note that not all metrics are
        valid with all algorithms.  Refer to the documentation of
        'sklearn.KDTree' for a description of available algorithms. 
        Default is 'euclidean'.
        
    leaf_size : int, default=40.
        Specify the leaf size of the underlying tree.  

    seed : int, default=1. 
        Determines random number for np.random.seed to generate
        random samples. Pass an int for reproducible results
        across multiple function calls.
        
    score_criterion: {"MISE", "KL"}, default="MISE".
        The non-parameteric criterion used for model selection. 
        See paper for details.
    
    sampling_stratigy: {"auto","bounded","normal","heavy_tail","mixed"}, 
        default="bounded".
        The inportance sampling scheme to estimate integration of AWNN.
        Use "bounded" if all entries are bounded. Use "normal" if data is 
        concentrated.Use "heavy_tail" or "mixed" if data is heavy tailed 
        but pay attention to numerical instability. See .utils for detail. 
    
   
    Attributes
    ----------
    n_train_ : int
        Number of training instances.

    tree_ : "KDTree" instance
        The tree algorithm for fast generalized N-point problems.

    dim_ : int
        Number of features.

    vol_unitball_ : float
        Volume of dim_ dimensional unit ball.

    score_validate_scale_: int, default= 2 * n_train_ * dim_.
        Number of points used to estimate integration of estimator.
        
    log_density: array-like of shape (n_test, ).
        Estimated log-density of test samples.
    

    See Also
    --------
    sklearn.neighbors.KDTree : K-dimensional tree for fast generalized N-point
        problems.
    
    Reference
    ---------
    Sanjoy Dasgupta and Samory Kpotufe. Optimal rates for k-nn density and mode 
    estimation. Advances in Neural Information Processing Systems, 27, 2014.
    
    Gérard Biau, Frédéric Chazal, David Cohen-Steiner, Luc Devroye, and Carlos 
    Rodríguez. A weighted k-nearest neighbor density estimate for geometric 
    inference. Electronic Journal of Statistics, 5(none):204 – 237, 2011. 
    doi: 10.1214/11-EJS606. URL https://doi.org/ 10.1214/11-EJS606.

    Puning Zhao and Lifeng Lai. Analysis of knn density estimation, 2020.

    Examples
    --------
    Compute a WKNN density estimate with a fixed k.

    >>> from KNN import KNN
    >>> import numpy as np
    >>> X_train = np.random.rand(2000).reshape(-1,2)
    >>> X_test = np.random.rand(6).reshape(-1,2)
    >>> WKNN_model = KNN(k=100).fit(X_train,method="KNN")
    >>> log_density = WKNN_model.score_samples(X_test)
    >>> log_density
    array([ 0.10936768, -0.04164363, -0.27935619])
    """

    def __init__(
                self,
                *,
                k=2,
                threshold_r=0.5,
                threshold_num=5,
                C=1,
                metric="euclidean",
                leaf_size=40,   
                seed=1,
                score_criterion="MISE",
                sampling_stratigy="bounded"
                
    ):
        
        self.k=k
        self.threshold_num=threshold_num
        self.threshold_r=threshold_r
        self.C=C
        self.metric = metric
        self.leaf_size=leaf_size
        self.seed=seed
        self.score_criterion=score_criterion
        self.sampling_stratigy=sampling_stratigy
        
        if metric not in KDTree.valid_metrics:
            raise ValueError("invalid metric: '{0}'".format(metric))
            
        self.log_density=None
        


    def fit(self, X, y=None,
            method="KNN",
            score_validate_scale="auto"):
        """Fit the AWNN on the data.

        Parameters
        ----------
        X : array-like of shape (n_train_, dim_)
            Array of dim_-dimensional data points.  Each row
            corresponds to a single data point.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
            
        method: {"KNN","WKNN","AKNN"}, default="KNN".
            NN method to use. 
            
        score_validate_scale: "auto" or int, default="auto".
            Inportance sampling scale. Set to 2*n_train_*dim if auto.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        
        self.tree_ = KDTree(
            X,
            metric=self.metric,
            leaf_size=self.leaf_size,
        )
        self.dim_=X.shape[1]
        self.n_train_=X.shape[0]
        self.vol_unitball_=math.pi**(self.dim_/2)/math.gamma(self.dim_/2+1)
        
        
        if score_validate_scale=="auto":
            self.score_validate_scale_=self.n_train_*(self.dim_*2)
        else:
            self.score_validate_scale_=score_validate_scale
        
        self.method_ = method
        
        if self.method_=="BKNN":
            if self.dim_==1:
                self.C2=np.sqrt(np.cov(X.T))
            else:
                self.C2=np.sqrt(np.linalg.det(np.cov(X.T)))
            self.C2=self.C2*(0.028*self.n_train_**(0.8))**(1/self.dim_)
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
        if self.method_ == "KNN":
            log_density=knn(X,self.tree_,self.k,self.n_train_,self.dim_,
                            self.vol_unitball_)
        elif self.method_ =="WKNN":
            
            log_density=wknn(X,self.tree_,self.k,self.n_train_,self.dim_,
                            self.vol_unitball_)
        elif self.method_ =="AKNN":
            kwargs={"threshold_num":self.threshold_num,
                    "threshold_r":self.threshold_r}
            log_density=aknn(X,self.tree_,self.k,self.n_train_,self.dim_,
                            self.vol_unitball_,**kwargs)
        elif self.method_ =="BKNN":
            kwargs={"kmax":int(self.n_train_**(1/2)),
                    "C2":self.C2,
                    "C":self.C}
            
            log_density=bknn(X,self.tree_,self.n_train_,self.dim_,
                            self.vol_unitball_,**kwargs)
            
        else:
            raise ValueError("invalid method: '{0}'".format(self.method_))
            
        
        return log_density
    
    
    def predict(self,X,y=None):
        """Compute as well as update the log-likelihood of each sample under 
        the model.

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
        
        self.log_density=self.score_samples(X)
        
        return self.log_density
    
    
    def compute_KL(self,X):
        """Compute the KL statistic.

        Parameters
        ----------
        X : array-like of shape (n_test, dim_)
            List of n_test-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        KL statistic : float
            Computed KL statistc. 
            
        Reference
        ---------
        J. S. Marron. A Comparison of Cross-Validation Techniques in Density 
        Estimation. The Annals of Statistics, 15(1):152 – 162, 1987. 
        doi: 10.1214/aos/1176350258. URL https: //doi.org/10.1214/aos/1176350258.
        """
        
        # Monte Carlo estimation of integral
        kwargs={"ruleout":0.01,"method":self.sampling_stratigy,"seed":self.seed}
        X_validate,pdf_X_validate=mc_sampling(X,nsample=self.score_validate_scale_,**kwargs)
        validate_log_density=self.score_samples(X_validate)
        
        # if density has been computed, then do not update object attribute
        if self.log_density is None:
            self.log_density=self.score_samples(X)
        
        return self.log_density.mean()-(np.exp(validate_log_density)/pdf_X_validate).mean()
    
        
    def compute_MISE(self,X):
        """Compute the MISE statistic.

        Parameters
        ----------
        X : array-like of shape (n_test, dim_)
            List of n_test-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        MISE statistic : float
            Computed MISE statistc. 
            
        Reference
        ---------
        Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. 
        Springer Publishing Company, Incorporated, 1st edition, 2008. 
        ISBN 0387790519.
        """
        
        # Monte Carlo estimation of integral
        kwargs={"ruleout":0.01,"method":self.sampling_stratigy,"seed":self.seed}
        X_validate,pdf_X_validate=mc_sampling(X,nsample=self.score_validate_scale_,**kwargs)
        validate_log_density=self.score_samples(X_validate)
        
        # if density has been computed, then do not update object attribute
        if self.log_density is None:
            self.log_density=self.score_samples(X)
        
        return 2*np.exp(self.log_density).mean()-(np.exp(2*validate_log_density)/pdf_X_validate).mean()
    
    def compute_ANLL(self,X):
        """Compute the average negative log-likelihood.

        Parameters
        ----------
        X : array-like of shape (n_test, dim_)
            List of n_test-dimensional data points.  Each row corresponds 
            to a single data point.

        Returns
        -------
        ANLL statistic : float
            Computed ANLL statistc. 
        """
        
        # if density has been computed, then do not update object attribute
        if self.log_density is None:
            self.log_density=self.score_samples(X)

        return -self.log_density.mean()

    def score(self, X, y=None):
        """Compute the total score under the model. Update average negative
        log-likelihood of test samples.

        Parameters
        ----------
        X : array-like of shape (n_test, dim_)
            List of n_test-dimensional data points.  Each row
            corresponds to a single data point.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        score : float
            Total score of the data in X. Computed via method in 
            score_criterion.
        """
        self.ANLL=self.compute_ANLL(X)
        
        if self.score_criterion=="KL":
            self.KL=self.compute_KL(X)
            return self.KL
        elif self.score_criterion=="MISE":
            self.MISE=self.compute_MISE(X)
            return self.MISE
          
                
                
            
                
        
        
        
    
    

    

