# Summary

    Adaptive Weighted Nearest Neighbor (**AWNN**) is an adaptive unsupervised algorithm for density estimation. Motivated by the bias-variance decomposition for the pointwise error of the weighted nearest neighbors for density estimation, we provide an efficient optimization approach to select the weights of nearest neighbors for each instance adaptively from the data. From a theoretical perspective, we establish the convergence rates of **AWNN** in terms of the $L_p$-norm when the marginal distribution $P$ has unbounded support, which coincides with the minimax lower bound. Moreover, we for the first time manage to explain the benefits of the adaptive method in density estimation. To be specific, we find that to achieve the optimal convergence rate, the condition that **AWNN** requires is weaker than that of the standard weighted $k$-NN density estimator. In addition, we show that the convergence rate of **AWNN** is no worse than that of the standard weighted $k$-NN density estimator. In the experiments, we verify the theoretical findings and show the superiority of **AWNN** through both synthetic and real-world data experiments.

# Statement of need

    A vast literature has focused on density estimation. For example, the most intuitive approach, histogram methods find partitions of input space and estimate density with bins [@lopez2013histogram]. Although histogram density estimation enjoys sound theoretical properties, it suffers from low efficiency and boundary discontinuity. Besides, it is sensitive to the choice of bin width. To overcome these issues, the kernel density estimation was proposed in [@rosenblatt1956remarks; @parzen1962estimation] with theoretical properties that are well explored. However, when encountering density distributions with varying local properties, the kernel density estimation will have a poor performance. To conquer the weakness, nearest neighbors-based methods for density estimation, proposed in [@loftsgaarden1965nonparametric], were investigated in [@biauweighted and @biaulecture] and successfully applied to many machine learning tasks, like density-based clustering or anomaly detection, see, e.g., [wu2019fast; @gu2019statistical; @zhang2021adaptive].

    There are several advantages of $k$-NN density estimation compared with other methods. First of all, $k$-NN is a lazy learning method that requires no training stage and has attractive testing stage sample complexity. Moreover, the smoothing of k-NN varies according to the numberof observations in a particular region, which can be regarded as a variant of the kernel density estimation with the local choice of the bandwidth [@orava2011k]. Furthermore, mild conditions are required about the underlying distribution to provide a convergence guarantee because of $k$-NN’s non-parametric instinct. We refer the reader to [@biaulecture] for more discussions. Recently, [@papernot2018deep); @gopfert2022interpretable] further pointed out that the straightforward logic of k-NN naturally satisfies the requirements of trustworthy AI.

# Methods

    Before we proceed, we need to introduce some basic notations. For any $x ∈ \mathbb{R}^d$, we denote $X_{(k)}(x):=X_{(k)}(x;D_n)$  as the $k$-th nearest neighbor of $x$ in $D_n$. Then we denote $R_k(x) := R_k(x; D_n)$ as the distance between $x$ and $X_{(k)} (x; D)$, termed as the $k$-distance of $x$ in $D_n$. Given $n$ independent identically distributed data $D_n \in \mathbb{R}^d$. For each point $x \in \mathbb{R}^d$, the $k$-NN density estimator [@loftsgaarden1965nonparametric; @dasgupta2014optimal], is defined as follows.

$$
f_k(x)=\frac{k/n}{V_d R_k^d(x)},
$$

  where $V_d$ be the volume of the unit ball in $\mathbb{R}^d$. An intuitive explanation of this estimator is that from order statistics [@david2004order]. We know that $P(B(x, R_k(x)))$ follows Beta Binomial distribution $Beta(k, n − k + 1)$. As a result, we have $\mathbb{E}[P(B(x, R_k(x)))] = k/(n + 1)$. Therefore, we have the approximation $k/n ≈ P(B(x, R_k(x))) ≈ f(x)V_d R_k(x)^d$. This yields the standard $k$-NN estimator (1).

# Reference

    
