# Toolbox
Small pieces of codes that I often use

# PairProbabilty
This code provides a way to approximate the probability of
finding two features together using von Neumann Diffusion Kernel.

Also plots a cluster heatmap of the normalized von Neumann diffusion matrix

### Required packages:
- Numpy
- Scipy
- Seaborn (Matplotlib, Pandas)

### How it works:
- make your dataset a comma separated file that has the columns as features and rows as samples
- include the feature names on top of each column
- then run:
python pairProbabilty.py {path/to/dataset}.csv
