"""
Adaptive Weighted Nearest Neighbor Density Estimation
-----------------------------------------------------
"""

import numpy as np
import math
from sklearn.neighbors import KDTree

from ._utils import mc_sampling
from ._weight_selection import weight_selection



class AWNN(object):
    """AWNN

    Read more in Adaptive Weighted Nearest Neighbor Density Estimation

    Parameters
    ----------
    C : float, default=1.0.
        The tuning parameter in AWNN which controls the optimized weights. 

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
    
    cut_off : int, default=5.
        Number of neighbors for cutting AWNN to KNN. 

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
        
    max_neighbors_: int, default= n_train_ ** 2/3.
        Maximum number of neighbors quried from KDTree. 

    score_validate_scale_: int, default= 2 * n_train_ * dim_.
        Number of points used to estimate integration of estimator.
        
    log_density: array-like of shape (n_test, ).
        Estimated log-density of test samples.
    
    estAlpha: array-like of shape (n_test, n_train_ ).
        Estimated weights of test samples.

    See Also
    --------
    sklearn.neighbors.KDTree : K-dimensional tree for fast generalized N-point
        problems.

    Examples
    --------
    Compute a AWNN density estimate with a fixed C.

    >>> from AWNN import AWNN
    >>> import numpy as np
    >>> X_train = np.random.rand(2000).reshape(-1,2)
    >>> X_test = np.random.rand(6).reshape(-1,2)
    >>> AWNN_model = AWNN(C=1).fit(X_train)
    >>> log_density,_ = AWNN_model.score_samples(X_test)
    >>> log_density
    array([ 0.10367955, -0.01632248,  0.06320222])
    """

    def __init__(
        self,
        *,
        C=1.0,
        metric="euclidean",
        leaf_size=40,
        seed=1,
        score_criterion="MISE",
        sampling_stratigy="bounded",
        cut_off=5
    ):
        self.C = C
        self.metric = metric
        self.leaf_size=leaf_size
        self.seed=seed
        self.score_criterion=score_criterion
        self.sampling_stratigy=sampling_stratigy
        self.cut_off=cut_off
        
        if metric not in KDTree.valid_metrics:
            raise ValueError("invalid metric: '{0}'".format(metric))
            
        self.log_density=None
        


    def fit(self, X, y=None,
            max_neighbors="auto",
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
            
        max_neighbors: "auto" or int, default="auto".
            Scale of first step query in AWNN for efficiency. Set to n**(2/3)
            if auto.
            
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
        
        if max_neighbors=="auto":
            # generally enough with 1e4
            self.max_neighbors_=min(int(X.shape[0]**(2/3)),10000)
        else:
            self.max_neighbors_=max_neighbors
            
        if score_validate_scale=="auto":
            self.score_validate_scale_=self.n_train_*(self.dim_*2)
        else:
            self.score_validate_scale_=score_validate_scale
        
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
        for key in ['C',"cut_off"]:
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
        
        n_test=X.shape[0]
        
        log_density=np.zeros(n_test)
        
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
                    distance_vec=distance_vec
                    beta=self.C*distance_vec
                    estAlpha,alphaIndexMax=weight_selection(beta,cut_off=self.cut_off)
                    
            density_num=np.array([k for k in range(1,alphaIndexMax+1)]).dot(estAlpha[:alphaIndexMax])
            density_den=np.array([distance_vec[:alphaIndexMax]**self.dim_]).dot(estAlpha[:alphaIndexMax])
            
            if density_num<=0 or density_den<=0:
                log_density[i]=-30
            else:
                log_density[i] = math.log(density_num)-math.log(density_den)
                
        log_density-=np.log(self.n_train_*self.vol_unitball_)
   
        return log_density, None
    
    
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
        
        estAlpha : ndarray of shape (n_test, n_train_)
            Estimated weights of test instances with respect to training 
            instances.
        """

        self.log_density,self.estAlpha=self.score_samples(X)
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
        kwargs={"ruleout":0.01,"method":self.sampling_stratigy,"seed":1}
        X_validate,pdf_X_validate=mc_sampling(X,nsample=self.score_validate_scale_,**kwargs)
        validate_log_density,_=self.score_samples(X_validate)
        
           
        # if density has been computed, then do not update object attribute
        if self.log_density is None:
            self.log_density,self.estAlpha=self.score_samples(X)
        
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
        kwargs={"ruleout":0.01,"method":self.sampling_stratigy,"seed":1}
        X_validate,pdf_X_validate=mc_sampling(X,nsample=self.score_validate_scale_,**kwargs)
        validate_log_density,_=self.score_samples(X_validate)

        # if density has been computed, then do not update object attribute
        if self.log_density is None:
            self.log_density,self.estAlpha=self.score_samples(X)
        
        return 2*np.exp(self.log_density).mean()-(np.exp(2*validate_log_density)/pdf_X_validate).mean()
    
    
    def compute_ANLL(self,X):
        """Compute the average negative log-likelihood.

        Parameters
        ----------
        X : array-like of shape (n_test, dim_)
            List of n_test-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        ANLL statistic : float
            Computed ANLL statistc. 
        """
        
        # if density has been computed, then do not update object attribute
        if self.log_density is None:
            self.log_density,self.estAlpha=self.score_samples(X)
        
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
        
                
                
            
                
        
        
        
    
    

    

