"FIX PROGRAM SETTINGS"
READ_DATASET = True
PLOT_PATH = "./figs/"
RS_PATH = "./results/"

N_clients = 50
DATASETS= ["mnist","fmnist"]
DATASET = DATASETS[0]

RUNNING_ALGS = ["fedavg","fedprox","demlearn","demlearn-p"]
RUNNING_ALG = RUNNING_ALGS[3]

### Agorithm Parameters ###
if(DATASET == "mnist"):
    NUM_GLOBAL_ITERS = 5
    PARAMS_mu = 0.002  # 0.005, 0.002, 0.001, 0.0005  => choose 0.002

elif(DATASET == "fmnist"):
    NUM_GLOBAL_ITERS = 100
    PARAMS_mu = 0.001  # 0.005, 0.002, 0.001, 0.0005  => select 0.001

if(N_clients == 100):
        PARAMS_mu = 0.0005

PARAMS_gamma = 1.
PARAMS_beta = 1.
DECAY = True # True or False   Decay of smooth update beta
K_Levels = 3  #1, 2, 3  # plus root level => K = K+1 in paper
TREE_UPDATE_PERIOD = 2 # tested with 1, 2, 3, NUM_GLOBAL_ITERS
CLUSTER_METHOD = "weight" #"weight" or "gradient"
MODEL_TYPE = "cnn" #"cnn" or "mclr"

if "dem" in RUNNING_ALG:
    rs_file_path= RS_PATH + "{}_{}_I{}_K{}_T{}_b{}_d{}_m{}_{}.h5".format(
        DATASET, RUNNING_ALG, NUM_GLOBAL_ITERS, K_Levels, TREE_UPDATE_PERIOD,
        str(PARAMS_beta).replace(".","-"), DECAY, str(PARAMS_mu).replace(".","-"), CLUSTER_METHOD[0])

    print("Result Path: ", rs_file_path)
else:
    rs_file_path = RS_PATH + "{}_{}_I{}.h5".format(DATASET, RUNNING_ALG, NUM_GLOBAL_ITERS)
    print("Result Path: ", rs_file_path)





