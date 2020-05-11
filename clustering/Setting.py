"FIX PROGRAM SETTINGS"
PLOT_PATH = "./figs/"
RS_PATH = "./results/"
K_Levels = 1  #2, 3
N_clients = 50
NUM_GLOBAL_ITERS = 100 #100
TREE_UPDATE_PERIOD = 2 # tested with 1, 2, 3
CLUSTER_METHOD = "weight"
DECAY = False # True or False   Decay of smooth update
# CLUSTER_METHOD = "gradient" #"gradient"  #"weight" or "gradient"
MODEL_TYPE = "cnn" #"cnn" or "mclr"
# RUNNING_ALG= "fedavg"
#RUNNING_ALG= "fedprox"
RUNNING_ALG = "demavg"
# RUNNING_ALG = "demprox"


