"FIX PROGRAM SETTINGS"
PLOT_PATH = "./figs/"
RS_PATH = "./results/"
# DATASET="cifar100" # "mnist" or "cifar100"
DATASET="mnist" # "mnist" or "cifar100"
K_Levels = 1  #2, 3
N_clients = 50
NUM_GLOBAL_ITERS = 60 #100
TREE_UPDATE_PERIOD = 2 # tested with 1, 2, 3
DECAY = True # True or False   Decay of smooth update
# CLUSTER_METHOD = "gradient"
CLUSTER_METHOD = "weight" #"weight" or "gradient"
# MODEL_TYPE = "cnn" #"cnn" or "mclr"
MODEL_TYPE = "mclr" #"cnn" or "mclr"
# RUNNING_ALG= "fedavg"
# RUNNING_ALG= "fedprox"

# RUNNING_ALG = "demlearn"
RUNNING_ALG = "demlearn-p"


