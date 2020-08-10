# Introduction

This repository contains the code to reproduce the experimental results in the paper ["Capacities and efficient computation of first-passage probabilities"](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.102.023304).

# Setting up the environment

1. Clone this repository
```
git clone https://github.com/StannisZhou/entropic_barrier.git
```

2. Set up the virtual environment
```
cd entropic_barrier
conda env create -f environment.yml
source activate entropic_barrier
python setup.py develop
```

# Reproduce the results

Run the three scripts under the `scripts` folder to reproduce the results in the paper.

To save time, a copy of the results from existing experiments is attached in the `results` folder. Use the `visualize_results.ipynb` notebook to visualize the results. 
