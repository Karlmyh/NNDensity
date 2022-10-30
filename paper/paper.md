# Summary

    Nearest Neighbor based Density Estimation is a class of density estimation method which improve the traditional kernel density estimation by allowing the estimation have varying bandwidth depending on nearest neighbor distances. Several advantages are possessed by NN based density estimations. They are lazy learning methods that require no training stage. They utilize local information for bandwidth selection [@orava2011k]. Their straightforward logic naturally satisfies the requirements of trustworthy AI [@papernot2018deep); @gopfert2022interpretable]. The NNDE package provide an efficient (numba based) implementation of six NN based density methods that users can directly apply in related studies. The package are presented in class-based for extensibility and is compatible with sklearn based parameter tuning functions such as GridSearchCV. NNDE's built-in cython implemented adaptive KD tree, which is modidied from sklearn, provides convinient local neighbor selection scheme and is extensible for more adaptive selection functions. Moreover, we provide efficient tools for complex distributions generation and density estimation visualization.

# Statement of Need



    There are barely any implementation of nearest neighbor based density estimation methods since the oridinal algorithm is, if possessed with well developed KD tree structure, straight forward and leaves no space for further optimizing. However, as [] illustrated, performance of estimation, such as accuracy and robustness, is improved if NN density estimation is weighted or adaptive. These evolution of NN brings challenge to algorithm implementation as well as parameter tuning. NNDE for the first time provides user friendly functions for six NN based density methods, namely KNN [@loftsgaarden1965nonparametric], WKNN [@biauweighted], TKNN [@zhao2020analysis], BKNN [@kovacs2017balanced] and newly proposed AKNN and AWNN. For parameter tuning, NNDE is compatible with cross validation methods in sklearn and is extensible for further development. Under research framework of density estimation, basic visualization tools that exhibit behavior of estimation in arbitray dimensions are provided. Also, NNDE include efficient functions for generating complex distributions such as densities with different marginals and mixture of densities. 

    A key component for NN based methods is the KD tree structure. A great many packages provided different implementation schemes [] for KD tree. Githubpage has done a thourough comparison from perspectives such as dimension restriction, query speed and parallelizability. However, consider if we want to search the largest $k$ such that $k\cdot R_k(x)<C$, where $R_k(x)$ is the kth nearest neighbor distance for $x$. For common KD tree implementations, we would have to query $(R_1(x),\cdots, R_n(x))$ and search iteratively. This guarantees the correct solution when $k=n-1$ but causes much waste of computation if $k=2$. Thus, NNDE choose to modify sklearn, which is based on cython, to implement a KD tree structure with built in adaptive neighbor search algorithm. The algorithm halts the searching of neighbors when nearest neighbor distances exceed some thresholds, which is a much more efficient approach than static searching after querying all the distances. 


# Methods

daptive Weighted Nearest Neighbor (**AWNN**) is an adaptive unsupervised algorithm for density estimation. Motivated by the bias-variance decomposition for the pointwise error of the weighted nearest neighbors for density estimation, we provide an efficient optimization approach to select the weights of nearest neighbors for each instance adaptively from the data. From a theoretical perspective, we establish the convergence rates of **AWNN** in terms of the $L_p$-norm when the marginal distribution $P$ has unbounded support, which coincides with the minimax lower bound. Moreover, we for the first time manage to explain the benefits of the adaptive method in density estimation. To be specific, we find that to achieve the optimal convergence rate, the condition that **AWNN** requires is weaker than that of the standard weighted $k$-NN density estimator. In addition, we show that the convergence rate of **AWNN** is no worse than that of the standard weighted $k$-NN density estimator. In the experiments, we verify the theoretical findings and show the superiority of **AWNN** through both synthetic and real-world data experiments.

    In this section, we introduce the nearest neighbor estimation methods included in NNDE. We denote $R_k(x)$ as the distance between $x$ and its kth nearest neighbor. Given $n$ independent identically distributed data $D_n \in \mathbb{R}^d$, the standard $k$-NN density estimator (**KNN**) [@loftsgaarden1965nonparametric; @dasgupta2014optimal] for each point $x \in \mathbb{R}^d$ is defined as 
$$
f_k(x)=\frac{k/n}{V_d R_k^d(x)}
$$
where $V_d$ be the volume of the unit ball in $\mathbb{R}^d$. An intuitive explanation of this estimator is that from order statistics [@david2004order]. We know that $P(B(x, R_k(x)))$ follows Beta Binomial distribution $Beta(k, n − k + 1)$. As a result, we have $\mathbb{E}[P(B(x, R_k(x)))] = k/(n + 1)$. Therefore, we have the approximation $k/n ≈ P(B(x, R_k(x))) ≈ f(x)V_d R_k(x)^d$. @biauweighted intended to smooth the standard KNN by aligned weighting and propsed **WKNN** defined as 
$$
{f}_{{W}}(x)=\frac{\sum_{i=1}^{k}w_{i}i/n}{V_d\sum_{i=1}^{k}w_{i} R_{i}^d(x)}
$$
where $w_1,\cdots w_k$ are fixed positive weights summing to 1. In practice, $w_i$ is usually set to $1/k$. 

KNN and WKNN both utilize fixed weights for all $x$. @zhao2020analysis proposed truncated method **TKNN** to fix the potential problem at the tail of distribution. They perform a pre-estimation using uniform kernel and substitute the pre-estimation with standard KNN when pre-estimation is large, namely 
$$
\hat{f}(\mathbf{x})=\left\{\begin{array}{cll}
\frac{k}{n V_d R_k^d(x) & \text { if } & \hat{f}(x) \leq a \\
\hat{f}(x) & \text { if } & \hat{f}(x)>a
\end{array}\right.
$$
where $\hat{f}(x)$ is uniform kernel density estimator. Their method is still non-adaptive, i.e. using fixed number of neighbors, in most regions. Recently, @kovacs2017balanced argued that adaptive choice of weights (number of neighbors) brings advantages to estimation when encountering densities with drastically varying value. **BKNN** [@kovacs2017balanced] demonstrate that by selecting largest $k$ such that
$$
k\cdot R_k(x)^d<C
$$
for some constant $C$ and dimension $d$, the prediction performance is better. However, choice of the statistic lacks theoretical support. Also, BKNN provide a plug-in parameter selection scheme for $C$, which is quit empirical and fails when dimension is higher than 2. A work in progress, called **AKNN**, argues that the suitable choice is instead 
$$
k\cdot R_k(x)^2<C. 
$$
while the selection of $C$ is done by cross validation. Both AKNN and BKNN is implemented through the built-in function in adaptive KD tree and achieve roughly the same computational cost as normal KD tree query. Moreover, another work in progress consider adaptive weighted nearest neighbor estimation. Weighted estimation for each $x$ is formalized as 
$$
	f_{A}(x)=\frac{\sum_{i=1}^{k(x)}w_{i}(x)i/n}{V_d\sum_{i=1}^{k(x)}w_{i}(x) R_{i}^d(x)}
$$
where $w_1(x),\cdots,w_{k(x)}(x)$ are local weights at $x$. Motivated by the bias-variance decomposition for the pointwise error, an efficient optimization approach is provided to select the weights of nearest neighbors for each $x$ adaptively. 

In what follows, we use a toy example to exhibits functionality of NNDE.  




# Reference

    
