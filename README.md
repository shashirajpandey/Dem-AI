# DemLearn Algorithm Implementation

We are working for the demonstration of some properties and initially develop new algorithm for Democratized Learning systems based on our Dem-AI philosophy paper
 
"Distributed and Democratized Learning: Philosophy and Research Challenges", [Paper Link](https://arxiv.org/abs/2003.09301)


This code is forked and reimplemented based on our prior paper and FedProx code:

"Federated Learning over Wireless Networks: Convergence Analysis and Resource Allocation"
[Paper Link](https://arxiv.org/abs/1910.13067)

FedProx implementation as: https://github.com/litian96/FedProx

Running Instruction
======
 In the current version, we support **MNIST** and **Fashion-MNIST** Dataset.
 
 **Environment:** Python 3.7, Tensorflow-gpu=1.15

 **Main File:** `dem_main.py`,  **Setting File:** `Seting.py`,  **Plot Summary Results:** `dem_plot_summary.py`, 
 
 **PATH:** Output: `./results`, Figures: `./figs`, Dataset: `./data`
 
### Input Parameters for the algorithm:
The more detail setting can be found in `Setting.py` file

- Four algorithms: *"fedavg","fedprox","demlearn","demlearn-p"*
- Datasets: *"mnist", "fmnist"*
- Clustering metrics: *"weight"*, *"gradient"*
- Model: *"cnn"*
- Result file name: *"{}_ {}_ I{}_ K{}_ T{}_ b{}_ d{}_ m{}_ {}.h5"* that reflects DATASET, RUNNING_ALG, NUM_GLOBAL_ITERS, K_Levels, TREE_UPDATE_PERIOD, beta, DECAY, mu, cluster_metric(w or g).
    * E.g.: `fmnist_demlearn_I100_K1_T2_b1-0_dTrue_m0-001_w.h5`, `mnist_demlearn-p_I60_K1_T60_b1-0_dTrue_m0-002_w.h5`
