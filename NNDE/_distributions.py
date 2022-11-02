'''
Distribution Generating Tools
-----------------
'''

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import beta
from scipy.stats import laplace
from scipy.stats import expon

from scipy.stats import cauchy
from scipy.stats import t


class Distribution(object): 
    """General Density Distribution Generation Objection
        """
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
    """MultivariateNormalDistribution
        Parameters
        ----------
        Mean : numpy.ndarray
            Mean of Multivariate Normal Distribution
        cov : numpy.ndarray
            covariance of Multivariate Normal Distribution
        dim : int
            dimension of data
        Examples
        --------
        generate data from Multivariate Normal Distribution
        >>> from _distributions import MultivariateNormalDistribution
        >>> import numpy as np
        >>> dim = 1
        >>> n_train = 5
        >>> density = MultivariateNormalDistribution(mean = np.zeros(dim)+1.5, cov = np.diag(np.ones(dim)*0.05))
        >>> X_train,pdf_X_train = density.generate(n_train)
        >>> X_train
        array([[1.63812078],
        [1.3961159 ],
        [0.87563862],
        [1.70910519],
        [1.15138903]])
        >>> pdf_X_train
        array([1.47425697, 1.60160866, 0.03617657, 1.15220536, 0.52921076])
        """
    def __init__(self, mean, cov):
        super(MultivariateNormalDistribution, self).__init__()
        self.mean = mean
        self.cov = cov
        self.dim = np.array([mean]).ravel().shape[0]
        
    def sampling(self, num_samples):
        """sample from this distribution
        Parameters
        ----------
        num_samples : int
            the size of the data sampled from multivariate normal distribution
        Returns
        -------
        params : numpy.ndarray
            data that sampled from multivariate normal distribution
        """
        return multivariate_normal.rvs(mean = self.mean,
                                       cov = self.cov,
                                       size = num_samples).reshape(-1,self.dim)

    def density(self, sample_X):
        """calculate the density of sample
        Parameters
        ----------
        sample_X : np.ndarray
            data that sampled from multivariate normal distribution
        Returns
        -------
        params : numpy.ndarray
            density of the data
        """
        return multivariate_normal.pdf(sample_X, mean = self.mean, cov = self.cov)



class UniformDistribution(Distribution): 
    """UniformDistribution
        Parameters
        ----------
        low : float
            the lower bound of the uniform distribution
        upper : float
            the upper bound of the uniform distribution
        dim : int
            dimension of data
        Examples
        --------
        generate data from uniform distribution
        >>> from _distributions import UniformDistribution
        >>> import numpy as np
        >>> dim = 1
        >>> n_train = 5
        >>> density = UniformDistribution(low = np.ones(dim)*2, upper = np.ones(dim)*4)
        >>> X_train,pdf_X_train = density.generate(n_train)
        >>> X_train
        array([[2.3242404 ],
        [2.85987217],
        [3.24476054],
        [2.19584214],
        [2.85021362]])
        >>> pdf_X_train
        array([0.5, 0.5, 0.5, 0.5, 0.5])
        """
    def __init__(self, low, upper):
        super(UniformDistribution, self).__init__()
        low = np.array(low).ravel()
        upper = np.array(upper).ravel()
        self.dim = len(low)
        self.low = low
        self.upper = upper
        
    def sampling(self, num_samples):
        """sample from this distribution
            Parameters
            ----------
            num_samples : int
                the size of the data sampled from uniform distribution
            Returns
            -------
            params : numpy.ndarray
                data that sampled from uniform distribution
        """
        return np.random.rand(num_samples,self.dim)*(self.upper-self.low)+self.low
    
    def density(self, sample_X):
        """calculate the density of sample
            Parameters
            ----------
            sample_X : np.ndarray
                data that sampled from uniform distribution
            Returns
            -------
            params : numpy.ndarray
                density of the data
            """
        in_interval_low = np.array([(sample_X[i] >= self.low).all() for i in range(len(sample_X))])
        in_interval_up = np.array([(sample_X[i] <= self.upper).all() for i in range(len(sample_X))])
        return in_interval_low*in_interval_up/np.prod(self.upper-self.low)



class LaplaceDistribution(Distribution):
    """LaplaceDistribution
        Parameters
        ----------
        loc : float
            Location parameters of the Laplace distribution
        scale : float
            Scale parameters of the Laplace distribution
        dim : int
            dimension of data
        Examples
        --------
        generate data from laplace distribution
        >>> from _distributions import LaplaceDistribution
        >>> import numpy as np
        >>> dim = 1
        >>> n_train = 5
        >>> density = LaplaceDistribution(scale = np.ones(dim)*0.5,loc = np.zeros(dim))
        >>> X_train,pdf_X_train = density.generate(n_train)
        >>> X_train
        array([[-0.00675993],
        [-1.04263469],
        [ 0.19283368],
        [ 0.25686772],
        [-0.06159066]])
        >>> pdf_X_train
        array([0.98657112, 0.12427364, 0.67999669, 0.59825664, 0.88410334])
        """
    def __init__(self, loc, scale):
        super(LaplaceDistribution, self).__init__()    
        self.loc = loc
        self.scale = scale
        self.dim = np.array(loc).ravel().shape[0]

    def sampling(self, num_samples):
        """sample from this distribution
            Parameters
            ----------
            num_samples : int
                the size of the data sampled from laplace distribution
            Returns
            -------
            params : numpy.ndarray
                data that sampled from laplace distribution
            """
        return np.random.laplace(loc = self.loc, scale = self.scale, size = (num_samples, self.dim))

    def density(self, sample_X):
        """calculate the density of sample
            Parameters
            ----------
            sample_X : np.ndarray
                data that sampled from laplace distribution
            Returns
            -------
            params : numpy.ndarray
                density of the data
            """
        return np.prod(laplace.pdf(sample_X, loc = self.loc, scale = self.scale) ,axis = 1 )



class BetaDistribution(Distribution):
    """BetaDistribution
        Parameters
        ----------
        a : float
            Parameter a of beta distribution
        b : float
            Parameter b of beta distribution
        dim : int
            dimension of data
        Examples
        --------
        generate data from beta distribution
        >>> from _distributions import BetaDistribution
        >>> import numpy as np
        >>> dim = 1
        >>> n_train = 5
        >>> density = BetaDistribution(a = np.ones(dim)*2,b = np.ones(dim)*2)
        >>> X_train,pdf_X_train = density.generate(n_train)
        >>> X_train
        array([[0.10566679],
        [0.44918474],
        [0.7819024 ],
        [0.40080439],
        [0.43581442]])
        >>> pdf_X_train
        array([0.56700793, 1.48450686, 1.02318621, 1.44096139, 1.47528127])
        """
    def __init__(self, a, b):
        super(BetaDistribution, self).__init__()
        self.a = a
        self.b = b
        self.dim = len(np.array(a).ravel())

    def sampling(self, num_samples):
        """sample from this distribution
            Parameters
            ----------
            num_samples : int
                the size of the data sampled from beta distribution
            Returns
            -------
            params : numpy.ndarray
                data that sampled from beta distribution
            """
        return np.random.beta(a = self.a, b = self.b, size = (num_samples, self.dim))

    def density(self, sample_X):
        """calculate the density of sample
            Parameters
            ----------
            sample_X : np.ndarray
                data that sampled from beta distribution
            Returns
            -------
            params : numpy.ndarray
                density of the data
            """
        return np.prod(beta.pdf(sample_X, a = self.a, b = self.b), axis = 1)
  


class ExponentialDistribution(Distribution):
    """ExponentialDistribution
            Parameters
            ----------
            lambda : float
                parameter lambda of exponential distribution
            dim : int
                dimension of data
            Examples
            --------
            generate data from beta distribution
            >>> from _distributions import ExponentialDistribution
            >>> import numpy as np
            >>> n_train = 5
            >>> density = ExponentialDistribution(lamda  = 0.5)
            >>> X_train,pdf_X_train = density.generate(n_train)
            >>> X_train
            array([[1.15586724],
            [1.20490937],
            [0.54141212],
            [1.51575227],
            [0.83933908]])
            >>> pdf_X_train
            array([0.19817847, 0.17966315, 0.67727556, 0.096486  , 0.37324099])
            """
    def __init__(self, lamda):
        super(ExponentialDistribution, self).__init__()
        self.dim = len(np.array(lamda).ravel())
        self.scale = lamda
        
    def sampling(self, num_samples):
        """sample from this distribution
            Parameters
            ----------
            num_samples : int
                the size of the data sampled from exponential distribution
            Returns
            -------
            params : numpy.ndarray
                data that sampled from exponential distribution
            """
        return np.random.exponential(scale = self.scale, size = (num_samples, self.dim))

    def density(self, sample_X):
        """calculate the density of sample
            Parameters
            ----------
            sample_X : np.ndarray
                data that sampled from exponential distribution
            Returns
            -------
            params : numpy.ndarray
                density of the data
            """
        return np.prod(expon.pdf(sample_X, scale = self.scale), axis = 1)
    


class MixedDistribution(Distribution):
    """MixedDistribution
        Parameters
        ----------
        density_seq : list
            densitiy_object of different distributions
        prob_seq : list
            probability of different distributions
        num_mix : int
            the number of distributions
        dim : int
            dimension of data
        Examples
        --------
        generate data from mixed distribution
        >>> from _distributions import MixedDistribution
        >>> import numpy as np
        >>> dim = 1
        >>> n_train = 5
        >>> density1 = MultivariateNormalDistribution(mean = np.zeros(dim)+1.5,cov = np.diag(np.ones(dim)*0.05))
        >>> density2 = MultivariateNormalDistribution(mean = np.zeros(dim)-1.5,cov = np.diag(np.ones(dim)*0.3))
        >>> density_seq = [density1, density2]
        >>> prob_seq = [0.4,0.6]
        >>> densitymix = MixedDistribution(density_seq, prob_seq)
        >>> X_train,pdf_X_train = densitymix.generate(n_train)
        >>> X_train
        array([[-1.91858336],
        [-1.47216895],
        [ 1.50752824],
        [-0.47449061],
        [-0.8211803 ]])
        >>> pdf_X_train
        array([0.32634577, 0.43645557, 0.71324543, 0.07573156, 0.20275207])
        """
    def __init__(self, density_seq, prob_seq):
        super(MixedDistribution, self).__init__()
        self.density_seq = density_seq
        self.prob_seq = prob_seq
        self.num_mix = len(density_seq)
        self.dim = self.density_seq[0].dim
        
    def sampling(self, num_samples):
        """sample from this distribution
            Parameters
            ----------
            num_samples : int
                the size of the data sampled from mixed distribution
            Returns
            -------
            params : numpy.ndarray
                data that sampled from mixed distribution
            """
        rd_idx = np.random.choice(self.num_mix,size = num_samples,
                                  replace = True, p = self.prob_seq)
        sample_X = []
        for i in range(self.num_mix):
            num_i = np.sum(rd_idx == i)
            density = self.density_seq[i]
            sample_Xi, _ = density.generate(num_i)
            sample_X.append(sample_Xi)
        sample_X = np.concatenate(sample_X, axis = 0)
        np.random.shuffle(sample_X)
        return sample_X

    def density(self, sample_X):
        """calculate the density of sample
            Parameters
            ----------
            sample_X : np.ndarray
                data that sampled from mixed distribution
            Returns
            -------
            params : numpy.ndarray
                density of the data
            """
        num_samples = sample_X.shape[0]
        pdf_true = np.zeros(num_samples, dtype = np.float64)
        for i in range(self.num_mix):
            prob = self.prob_seq[i]
            density = self.density_seq[i]
            pdf_true += prob * density.density(sample_X)
        return pdf_true


 
class TDistribution(Distribution):
    """TDistribution
        Parameters
        ----------
        scale : np.ndarray
            scale parameter of t-distribution
        loc : numpy.ndarray
            location parameter of t-distribution
        df : float
            degree of freedom of t-distruibution
        dim : int
            dimension of data
        Examples
        --------
        generate data from t-distribution
        >>> from _distributions import TDistribution
        >>> import numpy as np
        >>> dim = 1
        >>> n_train = 5
        >>> density = TDistribution(loc = np.zeros(dim),scale = np.ones(dim)*0.1,df = 2/3)
        >>> X_train,pdf_X_train = density.generate(n_train)
        >>> X_train
        array([[-1.67798488],
        [ 1.38915897],
        [ 1.5519168 ],
        [-1.36858926],
        [ 1.78834249]])
        >>> pdf_X_train
        array([0.41454429, 0.63114488, 0.69467125, 0.42462067, 0.31074675])
        """
    def __init__(self,scale,loc,df):
        super(TDistribution, self).__init__()
        self.scale = np.array(scale).ravel()
        self.loc = np.array(loc).ravel()
        self.df = df
        self.dim = len(np.array(scale).ravel())

    def sampling(self, num_samples):
        """sample from this distribution
            Parameters
            ----------
            num_samples : int
                the size of the data sampled from  t-distribution
            Returns
            -------
            params : numpy.ndarray
                data that sampled from t-distribution
            """
        sample_X = []
        for i in range(self.dim):
            sample_Xi = t.rvs(loc = self.loc[i],scale = self.scale[i],size = num_samples,df = self.df)
            sample_X.append(sample_Xi.reshape(-1,1))
        sample_X = np.concatenate(sample_X, axis = 1)
        return sample_X

    def density(self, sample_X):
        """calculate the density of sample
            Parameters
            ----------
            sample_X : np.ndarray
                data that sampled from t-distribution
            Returns
            -------
            params : numpy.ndarray
                density of the data
            """
        return np.prod(t.pdf(sample_X,loc = self.loc,scale = self.scale,df = self.df), axis = 1)
 


class CauchyDistribution(Distribution):
    """CauchyDistribution
        Parameters
        ----------
        scale : np.ndarray
            scale parameter of cauchy distribution
        loc : numpy.ndarray
            location parameter of cauchy distribution
        dim : int
            dimension of data
        Examples
        --------
        generate data from cauchy distribution
        >>> from _distributions import CauchyDistribution
        >>> import numpy as np
        >>> dim = 1
        >>> n_train = 5
        >>> density = CauchyDistribution(loc = np.zeros(dim),scale = np.ones(dim))
        >>> X_train,pdf_X_train = density.generate(n_train)
        >>> X_train
        array([[-1.57143399],
        [-1.43279245],
        [-0.67039748],
        [-3.03813366],
        [ 1.54796497]])
        >>> pdf_X_train
        array([0.43331842, 0.4337418 , 0.13878282, 0.00847301, 0.69741866])
        """
    def __init__(self,scale,loc):
        super(CauchyDistribution, self).__init__()
        self.scale = np.array(scale).ravel()
        self.loc = np.array(loc).ravel()
        self.dim = len(np.array(scale).ravel())

    def sampling(self, num_samples):
        """sample from this distribution
            Parameters
            ----------
            num_samples : int
                the size of the data sampled from  cauchy distribution
            Returns
            -------
            params : numpy.ndarray
                data that sampled from cauchy distribution
            """
        sample_X = []
        for i in range(self.dim):
            sample_Xi = cauchy.rvs(loc = self.loc[i],scale = self.scale[i],size = num_samples)
            sample_X.append(sample_Xi.reshape(-1,1))
        sample_X = np.concatenate(sample_X, axis=1)
        return sample_X

    def density(self, sample_X):
        """calculate the density of sample
            Parameters
            ----------
            sample_X : np.ndarray
                data that sampled from cauchy distribution
            Returns
            -------
            params : numpy.ndarray
                density of the data
            """
        return np.prod(cauchy.pdf(sample_X,loc = self.loc,scale = self.scale), axis = 1)
 


class MarginalDistribution(Distribution):
    """MarginalDistribution
            Parameters
            ----------
            density_object_vector : list
                density_object of different distributions
            dim_vector : numpy.array
                dimension of data of different distributions
            Examples
            --------
            generate data from cauchy distribution
            >>> from _distributions import LaplaceDistribution,UniformDistribution,MarginalDistribution
            >>> import numpy as np
            >>> dims = [1,1]
            >>> n_train = 5
            >>> density1 = LaplaceDistribution(scale = np.ones(dims[0])*0.5,loc = np.zeros(dims[0]))
            >>> density2 = UniformDistribution(low = np.ones(dims[1])*2,upper = np.ones(dims[1])*4)
            >>> density_seq = [density1, density2]
            >>> prob_seq = [1/2,1/2]
            >>> densitymix = MixedDistribution(density_seq, prob_seq)
            >>> marginal_density_vector = []
            >>> for i in range(dim): marginal_density_vector = marginal_density_vector+[densitymix]
            >>> densitymarginal = MarginalDistribution(marginal_density_vector)
            >>> X_train,pdf_X_train = densitymix.generate(n_train)
            >>> X_train
            array([[-1.14199292],
            [-2.03194307],
            [ 1.12048724],
            [-1.71588535],
            [-0.83467598]])
            >>> pdf_X_train
            array([0.29413559, 0.22724989, 0.21129491, 0.33696481, 0.17414632])
            """
    def __init__(self, density_object_vector):
        super(MarginalDistribution, self).__init__()   
        self.density_object_vector = density_object_vector
        self.dim_vector = np.array([],dtype='int32')
        for i in range(len(density_object_vector)):
            self.dim_vector = np.append(self.dim_vector,density_object_vector[i].dim)
            
    def sampling(self, num_samples):
        """sample from this distribution
            Parameters
            ----------
            num_samples : int
                the size of the data sampled from  marginal distribution
            Returns
            -------
            params : numpy.ndarray
                data that sampled from marginal distribution
            """
        sample_X = self.density_object_vector[0].sampling(num_samples)
        for i in range(1,len(self.dim_vector)):
            sample_X = np.hstack([sample_X,self.density_object_vector[i].sampling(num_samples)])
        return sample_X

    def density(self, sample_X):
        """calculate the density of sample
            Parameters
            ----------
            sample_X : np.ndarray
                data that sampled from marginal distribution
            Returns
            -------
            params : numpy.ndarray
                density of the data
            """
        pdf_true = np.ones(shape=sample_X.shape[0])
        for i in range(len(self.dim_vector)):
            dim_start = np.sum(self.dim_vector[range(i)])
            dim_end = np.sum(self.dim_vector[range(i)])+self.dim_vector[i]
            pdf_true *= self.density_object_vector[i].density(sample_X[:,range(dim_start,dim_end)])
        return pdf_true   
