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
        cv_sampling="auto",
        metric="euclidean",
        leaf_size=40,
        epsilon=1e-9,
        
    ):
        self.cv_sampling = cv_sampling
        self.C = C
        self.metric = metric
        self.leaf_size=leaf_size
        self.epsilon=epsilon
        
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
        self.estAlpha=np.zeros([n_test,self.max_neighbors])
        self.log_density=np.zeros(n_test)
        
        for i in range(n_test):
            distance_vec,_=self.tree_.query(X[i].reshape(1,-1),self.max_neighbors)
            distance_vec=distance_vec[distance_vec>0]
            beta=self.C*distance_vec
            
        
            self.estAlpha[i,:],alphaIndexMax=weight_selection(beta)
                
            density_num=np.array([k for k in range(1,alphaIndexMax+1)]).dot(self.estAlpha[i,:alphaIndexMax])
            density_den=np.array([distance_vec[:alphaIndexMax]**self.dim]).dot(self.estAlpha[i,:alphaIndexMax])
            
            if density_num==0:
                self.log_density[i]=np.log(self.epsilon)
            else:
                self.log_density[i] = np.log(density_num)-np.log(density_den)
            
        self.log_density-=np.log(self.n_train*self.vol_unitball)
            
        
        return self.log_density
    
    
    def predict(self,X,y=None):
        
        return self.score_samples(X)

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
        return self.log_density.mean()
    
    

    

