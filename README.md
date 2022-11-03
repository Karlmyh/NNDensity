# Nearest Neighbor Density Estimation (NNDensity)

The package implements six nearest neighbor based density estimation method and provides efficient tools for density estimation research. See paper/paper.md for more descriptions and details in methodology and literature.

## Contents

- [Installation](#Installation)
- [Basic Usage](#Basic-Usage)
  + [Data Generation](Data-Generation)
  + [Density Estimation](#Density-Estimation)
  + [Visualization](#Visualization)
- [Reference](Reference)

## Installation

Since *NNDensity* is based on *Cython*, installation requires c/c++ compiler. Users can check by 

```bash
gcc -v
g++ -v
```

to see their version. For Linux, users can install gcc/g++ by *apt*. For macOS, refer to *Xcode*. For Windows, refer to *Microsoft c++ building tools*. 

### Via PypI

```bash
pip install NNDensity
```

### Via GitHub

```bash
pip install git+https://github.com/Karlmyh/NNDensity.git
```


### Mannul Install
  > 
```bash
git clone git@github.com:Karlmyh/NNDensity.git
cd NNDensity 
python setup.py install
```


## Basic Usage

### Data Generation

Density generation tools. Below is a show case using a mixture distribution.

```python
from NNDensity import MultivariateNormalDistribution, MixedDistribution, ExponentialDistribution
# setup
dim=2
density1 = ExponentialDistribution(lamda = np.ones(dim)*0.5) 
density2 = MultivariateNormalDistribution(mean = np.zeros(dim)-1.5, cov = np.diag(np.ones(dim)*0.3)) 
density_seq = [density1, density2]
prob_seq = [0.4, 0.6]
densitymix = MixedDistribution(density_seq, prob_seq)

# generate 10 samples and return their pdf
samples, samples_pdf = densitymix.generate(10)
samples

# evaluate pdf at given samples
densitymix.density(samples)

# compare with true pdf
(samples_pdf == samples).all()
```
```python
Out[1]:  array([[-2.23087816, -1.08521314],
       [-1.03424594, -1.24327987],
       [-2.02698363, -1.63201056],
       [ 1.43021832,  1.51448518],
       [ 1.58820377,  1.8541296 ],
       [-0.88802267, -2.398429  ],
       [-1.26067249, -2.12988644],
       [-1.92476226, -2.0167295 ],
       [-2.0035588 , -1.35662414],
       [-1.46406062, -1.9693262 ]])
Out[2]: True
```



### Density Estimation

Adopt AWNN model to estimate the density. 

```python
###### using AWNN to estimate density
from NNDensity import AWNN

# generate samples
X_train, pdf_X_train =densitymix.generate(1000)
X_test, pdf_X_test =densitymix.generate(1000)

# choose parameter C=0.1
model_AWNN=AWNN(C=.1).fit(X_train)
# output is log scaled
est_AWNN=np.exp(model_AWNN.predict(X_test))
# compute the mean absolute error
np.abs(est_AWNN-pdf_X_test).mean()
```
```python
Out[3]:  0.09148487940943466
```

Automatically select parameter using *GridSearchCV* to improve result.

```python
from sklearn.model_selection import GridSearchCV

# generate samples
X_train, pdf_X_train =densitymix.generate(1000)
X_test, pdf_X_test =densitymix.generate(1000)

# select parameter grid
parameters={"k":[int(i*1000) for i in [0.01,0.02,0.05,0.1,0.2,0.5]]}
# use all available cpu, use 10 fold cross validation
cv_model_KNN=GridSearchCV(estimator=KNN(),param_grid=parameters,n_jobs=-1,cv=10)
_=cv_model_KNN.fit(X_train)
model_KNN=cv_model_KNN.best_estimator_
    
# output is log scaled
est_KNN=np.exp(model_KNN.predict(X_test))
# compute the mean absolute error
np.abs(est_KNN-pdf_X_test).mean()

```
```python
Out[4]:  0.055937476261628344
```




### Visualization

Frequently used visualization plots for density estimation research.

```python
###### 3d prediction surface using WKNN
from NNDensity import contour3d

# generate samples
dim=2
density1 = MultivariateNormalDistribution(mean = np.zeros(dim)+1.5, cov = np.diag(np.ones(dim)*0.4)) 
density2 = MultivariateNormalDistribution(mean = np.zeros(dim)-1.5, cov = np.diag(np.ones(dim)*0.7)) 
density_seq = [density1, density2]
prob_seq = [0.4, 0.6]
densitymix = MixedDistribution(density_seq, prob_seq)
X_train, pdf_X_train =densitymix.generate(1000)

model_plot=contour3d(X_train,method="WKNN",k=100)
model_plot.estimation()
fig=model_plot.make_plot()
```

<img src="https://github.com/Karlmyh/NNDensity/blob/main/paper/readme_example_1.png" width="300">




```python
###### 2d prediction contour using BKNN

from NNDensity import contour2d
from sklearn.model_selection import GridSearchCV

# generate samples
X_train, pdf_X_train =densitymix.generate(1000)

model_plot=contour2d(X_train,method="BKNN",C=10)
model_plot.estimation()
fig=model_plot.make_plot()
```

<img src="https://github.com/Karlmyh/NNDensity/blob/main/paper/readme_example_2.png" width="400">

```python
###### prediction curve plot

# generate samples
X_train, pdf_X_train =densitymix.generate(1000)


kargs_seq= [{"k":100},{"k":100},{"k":100} ]
model_plot=lineplot(X_train,method_seq=["KNN", "WKNN", "TKNN"],true_density_obj=densitymix,kargs_seq=kargs_seq)
fig=model_plot.plot()

kargs_seq= [{"C":0.9},{"C":1},{"C":1} ]
model_plot=lineplot(X_train,method_seq=["AKNN", "BKNN", "AWNN"],true_density_obj=densitymix,kargs_seq=kargs_seq)
fig=model_plot.plot()

```

<p float="left">
  <img src="https://github.com/Karlmyh/NNDensity/blob/main/paper/example_1.png" width="300" />
  <img src="https://github.com/Karlmyh/NNDensity/blob/main/paper/example_2.png" width="300" /> 
</p>








## Reference

*NNDensity* utilizes tools from *numpy*, *matplotlib*, *scipy*, *jupyter notebooks*, *scikit-learn*, *cython* and *numba*. Also, large part of KD tree implementation was borrowed from *scikit-learn*. For specific citations, see papers/paper.md. 

