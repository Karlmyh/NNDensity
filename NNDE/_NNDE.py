'''
Main Algorithm for NNDE Estimation
-----------------
'''


import numpy as np
import math
from sklearn.neighbors import KDTree
from ._kd_tree import KDTree as AKDTree
from ._utils import mc_sampling,weight_selection,knn,wknn,tknn,bknn,aknn




class NNDE(object):
    """General NNDE Density Estimation Object
    
        Parameters
        ----------
        metric : string , default = "euclidean"
            metric for the problem. For AKNN and BKNN, only euclidean supported
        leaf_size : int, default = 40
            leaf size used for KD tree
        seed : int,  default = 1
            random seed
        score_criterion : string default = "MISE"
            method used to cross validation
        sampling_stratigy : string default = "bounded"
            the sampling scheme used to estimate integral
        max_neighbors : int default = "auto"
            the maximum number of neighbors to use, set to max(10000, n**{2/3})
            if auto
        score_validate_scale : string default = "auto"
            sampling number, set to n*d**2 if auto
        
        Attributes
        ----------
        tree_ : sklearn KD tree or AKD tree object
        dim_ : dimension of data
        n_train_ : training sample size
        vol_unitball_ : volume of unit ball in dimension d space
        
    """
    def __init__(
        self,
        metric = "euclidean",
        leaf_size = 40,
        seed = 1,
        score_criterion = "MISE",
        sampling_stratigy = "bounded",
        max_neighbors = "auto",
        score_validate_scale = "auto"
    ):
        self.metric = metric
        self.leaf_size = leaf_size
        self.seed = seed
        self.score_criterion = score_criterion
        self.sampling_stratigy = sampling_stratigy
        self.max_neighbors = max_neighbors
        self.score_validate_scale = score_validate_scale
        if metric not in KDTree.valid_metrics:
            raise ValueError("invalid metric: '{0}'".format(metric))
        self.log_density = None

    def _fit(self, X, y=None):
        """Fitting the model using KD tree.

        Parameters
        ----------
        X : array-like of shape (n_train_, dim_)
            An array of points to train. 
            
        Returns
        -------
        self 
        
        """
        if self.max_neighbors == "auto":
            self.max_neighbors_ = min(int(X.shape[0]*(2/3)),10000)
        self.tree_ = KDTree(
            X,
            metric = self.metric,
            leaf_size = self.leaf_size,
        )
        self.dim_ = X.shape[1]
        self.n_train_ = X.shape[0]
        self.vol_unitball_ = math.pi**(self.dim_/2)/math.gamma(self.dim_/2+1)
        if self.score_validate_scale == "auto":
            self.score_validate_scale_ = self.n_train_*(self.dim_*2)
        return self
    
    def _adaptive_fit(self, X, y = None):
        """Fitting the model using Adaptive KD tree.

        Parameters
        ----------
        X : array-like of shape (n_train_, dim_)
            An array of points to train. 
            
        Returns
        -------
        self 
        
        """
        if self.max_neighbors == "auto":
            self.max_neighbors_ = min(int(X.shape[0]*(2/3)),10000)
        self.tree_ = AKDTree(
            X,
            metric = self.metric,
            leaf_size = self.leaf_size,
        )
        self.dim_ = X.shape[1]
        self.n_train_ = X.shape[0]
        self.vol_unitball_ = math.pi**(self.dim_/2)/math.gamma(self.dim_/2+1)
        if self.score_validate_scale == "auto":
            self.score_validate_scale_ = self.n_train_*(self.dim_*2)
        return self
    
    def get_params(self, deep = True):
        pass
    
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
        valid_params = self.get_params(deep = True)
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

    def predict(self,X,y = None):
        """Compute the log-likelihood of each sample under the model. 

        Parameters
        ----------
        X : array-like of shape (n_test, dim_)
            An array of points to query.  Last dimension should match dimension
            of training data (dim_).
            
        Attributes
        ----------
        log_denstiy : array-like of shape (n_test, )
            An array of log probability of query points.

        Returns
        -------
        log_density : ndarray of shape (n_test,)
            Log-likelihood of each sample in `X`.
        
        """
        self.log_density = self.score_samples(X)
        return self.log_density

    def compute_KL(self,X):
        """Compute the KL statistic of testing sample under the model. 

        Parameters
        ----------
        X : array-like of shape (n_test, dim_)
            An array of points to query.  Last dimension should match dimension
            of training data (dim_).
            
        Attributes
        ----------
        log_denstiy : array-like of shape (n_test, )
            An array of log probability of query points.

        Returns
        -------
        KL statistic : float
        
        """
        
        # Monte Carlo estimation of integral
        kwargs = {"ruleout":0.01,"method":self.sampling_stratigy,"seed":self.seed}
        X_validate,pdf_X_validate = mc_sampling(X,nsample = self.score_validate_scale_,**kwargs)
        validate_log_density = self.score_samples(X_validate)
        # if density has been computed, then do not update object attribute
        if self.log_density is None:
            self.log_density = self.score_samples(X)
        return self.log_density.mean()-(np.exp(validate_log_density)/pdf_X_validate).mean()

    def compute_MISE(self,X):
        """Compute the MISE statistic of testing sample under the model. 

        Parameters
        ----------
        X : array-like of shape (n_test, dim_)
            An array of points to query.  Last dimension should match dimension
            of training data (dim_).
            
        Attributes
        ----------
        log_denstiy : array-like of shape (n_test, )
            An array of log probability of query points.

        Returns
        -------
        MISE statistic : float
        
        """
        # Monte Carlo estimation of integral
        kwargs = {"ruleout":0.01,"method":self.sampling_stratigy,"seed":self.seed}
        X_validate,pdf_X_validate = mc_sampling(X,nsample = self.score_validate_scale_,**kwargs)
        validate_log_density = self.score_samples(X_validate)
        # if density has been computed, then do not update object attribute
        if self.log_density is None:
            self.log_density = self.score_samples(X)
        return 2*np.exp(self.log_density).mean()-(np.exp(2*validate_log_density)/pdf_X_validate).mean()
    
    def compute_ANLL(self,X):
        """Compute the ANLL statistic of testing sample under the model. 

        Parameters
        ----------
        X : array-like of shape (n_test, dim_)
            An array of points to query.  Last dimension should match dimension
            of training data (dim_).
            
        Attributes
        ----------
        log_denstiy : array-like of shape (n_test, )
            An array of log probability of query points.

        Returns
        -------
        ANLL statistic : float
        
        """
        # if density has been computed, then do not update object attribute
        if self.log_density is None:
            self.log_density = self.score_samples(X)
        return -self.log_density.mean()

    def score(self, X, y = None):
        """Compute the score statistic of testing sample under the model. 

        Parameters
        ----------
        X : array-like of shape (n_test, dim_)
            An array of points to query.  Last dimension should match dimension
            of training data (dim_).
            
        Attributes
        ----------
        log_denstiy : array-like of shape (n_test, )
            An array of log probability of query points.

        Returns
        -------
        score : float
        
        """
        self.ANLL = self.compute_ANLL(X)
        if self.score_criterion == "KL":
            self.KL = self.compute_KL(X)
            return self.KL
        elif self.score_criterion == "MISE":
            self.MISE = self.compute_MISE(X)
            return self.MISE
    
    

class AWNN(NNDE):
    """AWNN
        Read more in Adaptive Weighted Nearest Neighbor Density Estimation
        Parameters
        ----------
        C : float, default=1.0.
            The tuning parameter in AWNN which controls the optimized weights.

        cut_off : int, default=5.
            Number of neighbors for cutting AWNN to KNN.

        save_weights: boolean ,default=False,
            If Save the weights of model or not

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
    def __init__(self,
                 C = 1,
                 cut_off = 5,
                 save_weights = False,
                 ):
        super(AWNN, self).__init__()
        self.C = C
        self.cut_off = cut_off
        self.save_weights = save_weights

    def fit(self,X, y = None):
        return self._fit(X, y = None)

    def get_params(self, deep = True):
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
        n_test = X.shape[0]
        log_density = np.zeros(n_test)
        if self.save_weights:
            self.alpha = np.zeros((n_test,self.n_train_-1))
        for i in range(n_test):
            distance_vec,_ = self.tree_.query(X[i].reshape(1,-1),self.max_neighbors_+1)
            distance_vec = distance_vec[0]
            # rule out self testing
            if distance_vec[0] == 0:
                distance_vec = distance_vec[1:]
            else:
                distance_vec = distance_vec[:-1]
            beta = self.C*distance_vec
            estAlpha,alphaIndexMax = weight_selection(beta,cut_off = self.cut_off)
            # rule out self testing
            # query more points if all points are used
            if alphaIndexMax == self.max_neighbors_:
                distance_vec,_ = self.tree_.query(X[i].reshape(1,-1),self.n_train_)
                distance_vec = distance_vec[0]
                # rule out self testing
                if distance_vec[0] == 0:
                    distance_vec = distance_vec[1:]
                    beta = self.C*distance_vec
                    estAlpha,alphaIndexMax = weight_selection(beta,cut_off = self.cut_off)
                else:
                    distance_vec = distance_vec[:-1]
                    beta = self.C*distance_vec
                    estAlpha,alphaIndexMax = weight_selection(beta,cut_off = self.cut_off)
            if self.save_weights:        
                self.alpha[i,:estAlpha.shape[0]] = estAlpha
            density_num = np.array([k for k in range(1,alphaIndexMax+1)]).dot(estAlpha[:alphaIndexMax])
            density_den = np.array([distance_vec[:alphaIndexMax]**self.dim_]).dot(estAlpha[:alphaIndexMax])
            if density_num <= 0 or density_den <= 0:
                log_density[i] = -30
            else:
                log_density[i] = math.log(density_num)-math.log(density_den)
        log_density -= np.log(self.n_train_*self.vol_unitball_)
        return log_density
    


class KNN(NNDE):
    """KNN
        Read more in K   Nearest Neighbor Density Estimation
        Parameters
        ----------
        k : int, default=2
            Number of neighbors

        See Also
        --------
        sklearn.neighbors.KDTree : K-dimensional tree for fast generalized N-point
            problems.
        Examples
        --------
        Compute a KNN density estimate with a fixed K.
        >>> from AWNN import KNN
        >>> import numpy as np
        >>> X_train = np.random.rand(2000).reshape(-1,2)
        >>> X_test = np.random.rand(6).reshape(-1,2)
        >>> KNN_model = KNN(k=5).fit(X_train)
        >>> log_density,_ = KNN_model.score_samples(X_test)
        >>> log_density
        array([ 0.092323179, -0.03631248,  0.07321349])
        """
    def __init__(self,
                 k = 2):
        super(KNN, self).__init__()
        self.k = k
        
    def fit(self,X, y = None):
        return self._fit(X, y = None)
        
    def get_params(self, deep = True):
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
        for key in ['k']:
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def score_samples(self, X):
        log_density = knn(X,self.tree_,self.k,self.n_train_,self.dim_,
                         self.vol_unitball_)
        return log_density



class WKNN(NNDE):
    """WKNN
            Read more in K   Nearest Neighbor Density Estimation
            Parameters
            ----------
            k : int, default=2
                Number of neighbors

            See Also
            --------
            sklearn.neighbors.KDTree : K-dimensional tree for fast generalized N-point
                problems.
            Examples
            --------
            Compute a WKNN density estimate with a fixed K.
            >>> from AWNN import WKNN
            >>> import numpy as np
            >>> X_train = np.random.rand(2000).reshape(-1,2)
            >>> X_test = np.random.rand(6).reshape(-1,2)
            >>> WKNN_model = WKNN(C=1).fit(X_train)
            >>> log_density,_ = WKNN_model.score_samples(X_test)
            >>> log_density
            array([ 0.12865613, -0.00533184,  0.83342628])
            """
    def __init__(self,
                 k = 2):
        super(WKNN, self).__init__()
        self.k = k
        
    def fit(self,X, y = None):
        return self._fit(X, y = None)

    def get_params(self, deep = True):
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
        for key in ['k']:
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def score_samples(self, X):
        log_density = wknn(X,self.tree_,self.k,self.n_train_,self.dim_,
                         self.vol_unitball_)
        return log_density
    


class TKNN(NNDE):
    """TKNN
        Read more in Adaptive Weighted Nearest Neighbor Density Estimation
        Parameters
        ----------
        threshold_num : int default = 5
            ？
        threshold_r : float default = 0.5
            ？
        See Also
        --------
        sklearn.neighbors.KDTree : K-dimensional tree for fast generalized N-point
            problems.
        Examples
        --------
        Compute a TKNN density estimate with fixed threshold_num and threshold_r.
        >>> from AWNN import TKNN
        >>> import numpy as np
        >>> X_train = np.random.rand(2000).reshape(-1,2)
        >>> X_test = np.random.rand(6).reshape(-1,2)
        >>> TKNN_model = TKNN(threshold_num = 8, threshold_r = 1).fit(X_train)
        >>> log_density,_ = TKNN_model.score_samples(X_test)
        >>> log_density
        array([ -0.01331292, 0.09633435,  -0.05204863])
        """
    def __init__(self,
                 k=2,
                 threshold_num = 5,
                 threshold_r = 0.5,
                 ):
        super(TKNN, self).__init__()
        self.threshold_num = threshold_num
        self.threshold_r = threshold_r
        self.k=k
        
    def fit(self,X, y = None):
        return self._fit(X, y = None)

    def get_params(self, deep = True):
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
        for key in ["k",'threshold_r','threshold_num']:
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def score_samples(self, X):
        log_density = tknn(X,self.tree_,self.k,self.n_train_,self.dim_,
                        self.vol_unitball_,self.threshold_num,self.threshold_r)
        return log_density
    
    
    
class BKNN(NNDE):
    """BKNN
        Read more in Adaptive Weighted Nearest Neighbor Density Estimation
        Parameters
        ----------
        C : float, default=1.0.
            The tuning parameter in AWNN which controls the optimized weights.
        Attributes
        ----------
        C2 : float
            Thresholding constant in adaptive selection
        kmax : int
            Maximum number of neighbor to use
        Examples
        --------
        Compute a BKNN density estimate with a fixed C.
        >>> from AWNN import BKNN
        >>> import numpy as np
        >>> X_train = np.random.rand(2000).reshape(-1,2)
        >>> X_test = np.random.rand(6).reshape(-1,2)
        >>> BKNN_model = BKNN(C=1).fit(X_train)
        >>> log_density,_ = BKNN_model.score_samples(X_test)
        >>> log_density
        array([ 0.04913432, -0.04849256,  -0.01297819])
        """
    def __init__(self,
                 C = 0.5,
                 ):
        super(BKNN, self).__init__()
        self.C = C
        
    def fit(self,X, y = None):
        return self._adaptive_fit(X, y = None)

    def get_params(self, deep = True):
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

    def score_samples(self, X):
        if self.dim_ == 1:
            self.C2 = np.sqrt(np.cov(X.T))
        else:
            self.C2 = np.sqrt(np.linalg.det(np.cov(X.T)))
        self.C2 = self.C2*(0.028*self.n_train_**(0.8))**(1/self.dim_)
        self.kmax = int(self.n_train_**0.5)
        log_density = bknn(X,self.tree_,self.n_train_,self.dim_,
                        self.vol_unitball_,self.kmax,self.C,self.C2)
        return log_density
    


class AKNN(NNDE):
    """AKNN
        Read more in Adaptive Weighted Nearest Neighbor Density Estimation
        Parameters
        ----------
        C : float, default=1.0.
            The tuning parameter in AWNN which controls the optimized weights.
        Attributes
        ----------
        kmax : int
            Maximum number of neighbors to use
       
        Examples
        --------
        Compute a AKNN density estimate with a fixed C.
        >>> from AWNN import AKNN
        >>> import numpy as np
        >>> X_train = np.random.rand(2000).reshape(-1,2)
        >>> X_test = np.random.rand(6).reshape(-1,2)
        >>> AKNN_model = AKNN(C=1).fit(X_train)
        >>> log_density,_ = AKNN_model.score_samples(X_test)
        >>> log_density
        array([ 0.00089412, -0.05667348,  0.06320222])
        """
    def __init__(self,
                 C = 0.5,
                 ):
        super(AKNN, self).__init__()
        self.C = C
        
    def fit(self,X, y = None):
        return self._adaptive_fit(X, y = None)

    def get_params(self, deep = True):
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
        for key in ['C',"beta"]:
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out
    
    def score_samples(self, X):
        
        self.kmax = int(self.n_train_**0.5)
        log_density = aknn(X,self.tree_,self.n_train_,self.dim_,
                        self.vol_unitball_,self.kmax,self.C)
        return log_density






    