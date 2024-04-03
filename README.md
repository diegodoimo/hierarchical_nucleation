# Hierarchical nucleation in deep neural networks

This repository contains the code associated to the paper [Hierarchical nucleation in deep neural networks](https://arxiv.org/abs/2007.03506).

## Requirements
Please make sure to have Anaconda downloaded and installed

https://docs.conda.io/projects/conda/en/latest/user-guide/install/

To create a local enviroment with the packages required to run the code download this reporitory in an empty directory then type:

```setup
conda env create -f environment.yml
```

<!-- The specific requirements we used to run the code are: python 3.8\ cython 0.29\ numpy 1.18\ matplotlib 3.1\ scipy 1.4\ scikt-learn 0.22\ jupyter notebook -->

Once the environment is created to activate it type:

```setup
conda activate hier_nucl
```
Then open a jupyter notebook and follow the instructions to run the experiments described in the [paper](https://arxiv.org/abs/2007.03506).

```setup
jupyter notebook hier_nucl Hierarchical_nucleation_in_deep_networks.ipynb
```

To deactivate the environment type:

```setup
conda deativate
```
