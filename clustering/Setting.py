"FIX PROGRAM SETTINGS"
PLOT_PATH = "./figs/"
RS_PATH = "./results/"
# DATASET="cifar100" # "mnist" or "cifar100"
DATASET="mnist" # "mnist" or "cifar100"
K_Levels = 2  #2, 3
N_clients = 20
NUM_GLOBAL_ITERS = 50 #100
TREE_UPDATE_PERIOD = 1 # tested with 1, 2, 3
CLUSTER_METHOD = "gradient"
DECAY = False # True or False   Decay of smooth update
# CLUSTER_METHOD = "weight" #"gradient"  #"weight" or "gradient"
# MODEL_TYPE = "cnn" #"cnn" or "mclr"
MODEL_TYPE = "mclr" #"cnn" or "mclr"
# RUNNING_ALG= "fedavg"
# RUNNING_ALG= "fedprox"
RUNNING_ALG = "demavg"
# RUNNING_ALG = "demprox"


