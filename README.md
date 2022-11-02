# Nearest Neighbor Density Estimation (NNDE)

The package implements six nearest neighbor based density estimation method and provides efficient tools for density estimation research. See paper/paper.md for more descriptions and details in methodology and literature.

## Contents

- [Installation](#Installation)
- [Basic Usage](#Basic-Usage)
  + [Data Generation](Data-Generation)
  + [Density Estimation](#Density-Estimation)
  + [Visualization](#Visualization)
- [Reference](Reference)

## Installation

Since *NNDE* is based on *Cython*, installation requires c/c++ compiler. Users can check by 

```bash
gcc -v
g++ -v
```

to see their version. For Linux, users can install by *apt*. For macOS, refer to *Xcode*. For Windows, refer to *Microsoft c++ building tools*. 

### Via PypI

```bash
pip install NNDE
```

### Via GitHub

```bash
pip install git+https://github.com/Karlmyh/NNDE.git
```


### Mannul Install
  > 
```bash
git clone git@github.com:Karlmyh/NNDE.git
cd NNDE 
python setup.py install
```


## Basic Usage

### Data Generation

Density generation tools. Below is a show case using a mixture distribution.

```python
from NNDE import MultivariateNormalDistribution, MixedDistribution, ExponentialDistribution
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
samples_pdf
```
```python
array([[-2.23087816, -1.08521314],
       [-1.03424594, -1.24327987],
       [-2.02698363, -1.63201056],
       [ 1.43021832,  1.51448518],
       [ 1.58820377,  1.8541296 ],
       [-0.88802267, -2.398429  ],
       [-1.26067249, -2.12988644],
       [-1.92476226, -2.0167295 ],
       [-2.0035588 , -1.35662414],
       [-1.46406062, -1.9693262 ]])
1
```



### Density Estimation

### Visualization






## Reference
