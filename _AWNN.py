'''
Adaptive Weighted Nearest Neighbor
'''

import numpy as np
import math

from ._utils import grid_sampling
from sklearn.neighbors import KDTree
from ._weight_selection import weight_selection




# TODO: implement a brute force version for testing purposes
# TODO: bandwidth estimation
# TODO: create a density estimation base class?
class AWNN(object):
    """Kernel Density Estimation.

    Read more in the :ref:`User Guide <kernel_density>`.

    Parameters
    ----------
    bandwidth : float or {"scott", "silverman"}, default=1.0
        The bandwidth of the kernel. If bandwidth is a float, it defines the
        bandwidth of the kernel. If bandwidth is a string, one of the estimation
        methods is implemented.

    algorithm : {'kd_tree', 'ball_tree', 'auto'}, default='auto'
        The tree algorithm to use.

    kernel : {'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', \
                 'cosine'}, default='gaussian'
        The kernel to use.

    metric : str, default='euclidean'
        The distance metric to use.  Note that not all metrics are
        valid with all algorithms.  Refer to the documentation of
        :class:`BallTree` and :class:`KDTree` for a description of
        available algorithms.  Note that the normalization of the density
        output is correct only for the Euclidean distance metric. Default
        is 'euclidean'.

    atol : float, default=0
        The desired absolute tolerance of the result.  A larger tolerance will
        generally lead to faster execution.

    rtol : float, default=0
        The desired relative tolerance of the result.  A larger tolerance will
        generally lead to faster execution.

    breadth_first : bool, default=True
        If true (default), use a breadth-first approach to the problem.
        Otherwise use a depth-first approach.

    leaf_size : int, default=40
        Specify the leaf size of the underlying tree.  See :class:`BallTree`
        or :class:`KDTree` for details.

    metric_params : dict, default=None
        Additional parameters to be passed to the tree for use with the
        metric.  For more information, see the documentation of
        :class:`BallTree` or :class:`KDTree`.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    tree_ : ``BinaryTree`` instance
        The tree algorithm for fast generalized N-point problems.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    bandwidth_ : float
        Value of the bandwidth, given directly by the bandwidth parameter or
        estimated using the 'scott' or 'silvermann' method.

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.neighbors.KDTree : K-dimensional tree for fast generalized N-point
        problems.
    sklearn.neighbors.BallTree : Ball tree for fast generalized N-point
        problems.

    Examples
    --------
    Compute a gaussian kernel density estimate with a fixed bandwidth.

    >>> from sklearn.neighbors import KernelDensity
    >>> import numpy as np
    >>> rng = np.random.RandomState(42)
    >>> X = rng.random_sample((100, 3))
    >>> kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
    >>> log_density = kde.score_samples(X[:3])
    >>> log_density
    array([-1.52955942, -1.51462041, -1.60244657])
    """

    def __init__(
        self,
        *,
        C=1.0,
        metric="euclidean",
        leaf_size=40,
        epsilon=1e-9,
        seed=1
        
    ):
        self.C = C
        self.metric = metric
        self.leaf_size=leaf_size
        self.epsilon=epsilon
        self.seed=seed
        
        if metric not in KDTree.valid_metrics:
            raise ValueError("invalid metric: '{0}'".format(metric))
        


    def fit(self, X, y=None):
        """Fit the AWNN on the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        

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
        self.dim=X.shape[1]
        self.n_train=X.shape[0]
        self.vol_unitball=math.pi**(self.dim/2)/math.gamma(self.dim/2+1)
        self.max_neighbors=int(X.shape[0]**(2/3))
        self.score_validate_scale=self.n_train*self.dim**2
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
        for key in ['C']:
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
        X : array-like of shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).

        Returns
        -------
        density : ndarray of shape (n_samples,)
            Log-likelihood of each sample in `X`. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        n_test=X.shape[0]
        
        
        # est weight
        estAlpha=np.zeros([n_test,self.max_neighbors])
        log_density=np.zeros(n_test)
        
        for i in range(n_test):
            distance_vec,_=self.tree_.query(X[i].reshape(1,-1),self.max_neighbors+1)
            distance_vec=distance_vec[0]
            if distance_vec[0]==0:
                distance_vec=distance_vec[1:]
            else:
                distance_vec=distance_vec[:-1]
            beta=self.C*distance_vec
            
            
        
            estAlpha[i,:],alphaIndexMax=weight_selection(beta)
                
            density_num=np.array([k for k in range(1,alphaIndexMax+1)]).dot(estAlpha[i,:alphaIndexMax])
            density_den=np.array([distance_vec[:alphaIndexMax]**self.dim]).dot(estAlpha[i,:alphaIndexMax])
            
            if density_num==0:
                log_density[i]=np.log(self.epsilon)
            else:
                log_density[i] = np.log(density_num)-np.log(density_den)
            
        log_density-=np.log(self.n_train*self.vol_unitball)
            
        
        return log_density,estAlpha
    
    
    def predict(self,X,y=None):
        self.log_density,self.estAlpha=self.score_samples(X)
        return self.log_density,self.estAlpha
    
    
    def compute_KL(self,X):
        """Compute the KL statistic of given parameter.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        logprob : float
            Total log-likelihood of the data in X. This is normalized to be a
            probability density, so the value will be low for high-dimensional
            data.
        """
        X_validate,mass=grid_sampling(X,nsample=self.score_validate_scale,seed=self.seed)
        validate_log_density,_=self.score_samples(X_validate)
        
        
        '''
        int_est=0
        
        for i in range(self.score_validate_scale):
            distance_vec,_=self.tree_.query(X_validate[i].reshape(1,-1),self.max_neighbors)
            beta=self.C*distance_vec[0]
            #print(beta)
    
            estAlpha,alphaIndexMax=weight_selection(beta)
                
            density_num=np.array([k for k in range(1,alphaIndexMax+1)]).dot(estAlpha[:alphaIndexMax])
            density_den=np.array([distance_vec[0][:alphaIndexMax]**self.dim]).dot(estAlpha[:alphaIndexMax])
            
            if density_num==0:
                pass
            else:
                int_est+=density_num/density_den[0]/self.n_train/self.vol_unitball
                
           ''' 
        log_density,_=self.score_samples(X)
        
                
        return log_density.mean()-mass*np.exp(validate_log_density).mean()

    def score(self, X, y=None):
        """Compute the total log-likelihood under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        logprob : float
            Total log-likelihood of the data in X. This is normalized to be a
            probability density, so the value will be low for high-dimensional
            data.
        """
        
        return self.compute_KL(X)
                
                
            
                
        
        
        
    
    

    

