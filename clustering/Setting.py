"FIX PROGRAM SETTINGS"
PLOT_PATH = "./figs/"
K_Levels = 3  #2, 3
N_clients = 50
NUM_GLOBAL_ITERS = 10 #100
TREE_UPDATE_PERIOD = 2 # tested with 1, 2, 3
CLUSTER_METHOD = "weight"
# CLUSTER_METHOD = "gradient" #"gradient"  #"weight" or "gradient"
MODEL_TYPE = "cnn" #"cnn" or "mclr"
# RUNNING_ALG= "fedavg"
# RUNNING_ALG= "fedprox"
RUNNING_ALG = "demavg"
# RUNNING_ALG = "demprox"


