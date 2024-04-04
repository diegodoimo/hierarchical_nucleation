# Hierarchical nucleation in deep neural networks

This repository contains the code to reproduce the results of the paper [Hierarchical nucleation in deep neural networks](https://arxiv.org/abs/2007.03506). This work has been included in the [NeurIPS 2020 Proceedings](https://papers.nips.cc/paper_files/paper/2020/hash/54f3bc04830d762a3b56a789b6ff62df-Abstract.html).

### Requirements
We will use Anaconda to create the environment. You can install it by following:

https://docs.conda.io/projects/conda/en/latest/user-guide/install/

You can create the environment with the following command:

```setup
conda env create -f environment.yml
```

<!-- The specific requirements we used to run the code are: python 3.8\ cython 0.29\ numpy 1.18\ matplotlib 3.1\ scipy 1.4\ scikt-learn 0.22\ jupyter notebook -->

<br>

### Reproduce the paper plots.
You can reproduce the paper results following the examples in the jupyter notebook:

```setup
conda activate hier_nucl
jupyter notebook hier_nucl Hierarchical_nucleation_in_deep_networks.ipynb
```
