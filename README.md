# Introduction

This repository contains the code to reproduce the experimental results in the paper "Overcoming Entropic Barriers by Capacity Hopping".

# Setting up the environment

1. Clone this repository
```
git clone https://github.com/StannisZhou/capacity_hopping.git
```

2. Set up the virtual environment
```
cd capacity_hopping
conda create -n capacity_hopping --file conda_requirements.txt
source activate capacity_hopping
pip install -r pip_requirements.txt
python setup.py develop
```

# Reproduce the results

Run
```
python run_experiments.py
```
to run all the experiments. After you finish running all the experiments (which would take a while), run
```
python reproduce_results.py
```
to reproduce the results in the paper. This would generate a `report.pdf` file, which contains all the figures and numbers reported in the paper.
