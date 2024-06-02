# Black-box Detection of Backdoor Attacks with Limited Information and Data

This repo is to reproduce the experiment of the paper, [Black-box Detection of Backdoor Attacks with Limited Information and Data](https://arxiv.org/abs/2103.13127), based on PyTorch.

*I am not the author of the paper.

## Run

Make sure you install the [miniconda](https://docs.anaconda.com/free/miniconda/).

```
conda env create -f enviroment.yml
conda activate b3d
python -m attack.train
python -m defender.main
```
