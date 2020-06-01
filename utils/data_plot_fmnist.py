import h5py as hf
import os
import numpy as np
from clustering.Setting import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram
import matplotlib.gridspec as gridspec
plt.rcParams.update({'font.size': 16})  #font size 10 12 14 16 main 16
plt.rcParams['lines.linewidth'] = 2
XLim=100
YLim=0.1
#Global variable
markers_on = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90] #maker only at x = markers_on[]
RS_PATH = "./results/50users/" #directory of results (output of the system)
PLOT_PATH = "./figs/" #ploting results directory
# RS_PATH = "./results/50users/100iters"
OUT_TYPE = ".pdf" #.eps or .pdf #output figure type

#name of the output files. Very large output number, so change the file name acording to the output file name
name = {
        "avg1w": "demlearn_iter_100_k_1_w.h5",
        "avg1wf": "demlearn_iter_100_k_1_w_fixed.h5",
        "avg2w": "demlearn_iter_100_k_2_w.h5",
        "avg3g": "demlearn_iter_100_k_3_g.h5",
        "avg3w": "demlearn_iter_100_k_3_w.h5",
        "avg3wf": "demlearn_iter_100_k_3_w_fixed.h5",
        "prox1w": "demlearn-p_iter_100_k_1_w.h5",
        "prox1wf": "demlearn-p_iter_100_k_1_w_fixed.h5",
        "prox2w": "demlearn-p_iter_100_k_2_w.h5",
        "prox3w": "demlearn-p_iter_100_k_3_w_mu_001.h5",
        "prox3g": "demlearn-p_iter_100_k_3_g.h5",
        "prox3wf": "demlearn-p_iter_100_k_3_w_fixed.h5",
        "fedavg": "fedavg_iter_100.h5",
        "fedprox": "fedprox_iter_100.h5",
        "avg3b08": "demlearn_iter_100_k_3_w_beta_0_8.h5",
        # "avg3wdecay": "demlearn_iter_60_k_3_w_decay.h5",
        "avg3wg08": "demlearn_iter_100_k_3_w_gamma_0_8.h5",
        "avg3g1": "demlearn_iter_100_k_3_w_gamma_1.h5",
        "prox3wg08": "demlearn-p_iter_100_k_3_w_gamma_0_8.h5",
        "prox3wg1": "demlearn-p_iter_100_k_3_w_gamma_1.h5",
        "prox3wmu001": "demlearn-p_iter_100_k_3_w_mu_001.h5",
        "prox3wmu002": "demlearn-p_iter_100_k_3_w_mu_002.h5",
        "prox3wmu0005": "demlearn-p_iter_100_k_3_w_mu_0005.h5",
        "prox3wmu005": "demlearn-p_iter_100_k_3_w_mu_005.h5"
    }

color = {
    "gen": "royalblue",
    "cspe": "forestgreen",
    "cgen": "red",
    "c": "cyan",
    "gspe": "darkorange",  #magenta
    "gg": "yellow",
    "ggen": "darkviolet",
    "w": "white"
}
marker = {
    "gen": "8",
    "gspe": "s",
    "ggen": "P",
    "cspe": "p",
    "cgen": "*"
}

def  write_file(file_name = "../results/untitled.h5", **kwargs):
    with hf.File(file_name, "w") as data_file:
        for key, value in kwargs.items():
            #print("%s == %s" % (key, value))
            data_file.create_dataset(key, data=value)
    print("Successfully save to file!")
def read_data(file_name = "../results/untitled.h5"):
    dic_data = {}
    with hf.File(file_name, "r") as f:
        # List all groups
        #print("Keys: %s" % f.keys())
        for key in f.keys():
            dic_data[key] = f[key][:]
    return  dic_data



def plot_dendrogram(rs_linkage_matrix, round, alg):
    # Plot the corresponding dendrogram
    # change p value to 5 if we want to get 5 levels
    #dendogram supporting plot inside buitlin function, so we dont need to create our own method to plot figure or results
    plt.title('#Round=%s'%(round))

    rs_dendrogram = dendrogram(rs_linkage_matrix, truncate_mode='level', p=K_Levels)
    if(MODEL_TYPE == "cnn"):
        if(CLUSTER_METHOD == "gradient"):
            plt.ylim(0, 0.0006)
        else:
            plt.ylim(0, 1.)
    else:
        plt.ylim(0,1.5)

def plot_dendo_data_dem(file_name):
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(15, 6.2))
    num_plots = [241, 242, 243, 244, 245, 246, 247, 248]
    f_data = read_data(RS_PATH+name[file_name])
    dendo_data = f_data['dendo_data']
    dendo_data_round = f_data['dendo_data_round']
    print(dendo_data_round)
    i = 0
    for i in range(8):
        plt.subplot(num_plots[i])
        plot_dendrogram(dendo_data[i*7], dendo_data_round[i*7], RUNNING_ALG)
    plt.tight_layout()
    plt.savefig(PLOT_PATH+"den_" + file_name+OUT_TYPE)
    return 0

def plot_from_file():
    if("dem" in RUNNING_ALG):
        if(CLUSTER_METHOD == "weight"):
            file_name = RS_PATH+"{}_iter_{}_k_{}_w.h5".format(RUNNING_ALG, NUM_GLOBAL_ITERS, K_Levels)
        else:
            file_name = RS_PATH+"{}_iter_{}_k_{}_g.h5".format(RUNNING_ALG, NUM_GLOBAL_ITERS, K_Levels)
        f_data = read_data(file_name)
        TREE_UPDATE_PERIOD = f_data['TREE_UPDATE_PERIOD'][0]
        N_clients = f_data['N_clients'][0]
        ### PLOT DENDROGRAM ####
        dendo_data = f_data['dendo_data']
        dendo_data_round = f_data['dendo_data_round']
        i=0
        for m_linkage in dendo_data:
            plot_dendrogram(m_linkage, dendo_data_round[i], RUNNING_ALG)
            i+=1
    else:
        file_name = RS_PATH+"{}_iter_{}.h5".format(RUNNING_ALG, NUM_GLOBAL_ITERS)
        f_data = read_data(file_name)
        N_clients = f_data['N_clients'][0]

    print("DEM-AI --------->>>>> Plotting")


    alg_name = RUNNING_ALG+ "_"


    plt.figure(4)
    plt.clf()
    plt.plot(f_data['root_test'], label="Root_test", linestyle="--")
    if("dem" in RUNNING_ALG):
        # for k in range (K_Levels):
            plt.plot(f_data['gs_level_test'][-2,:,0], label="Gr(K)_spec_test", linestyle="-.")
            plt.plot(f_data['gg_level_test'][-2,:,0], label="Gr(K)_gen_test", linestyle="-.")


    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], label="Client_spec_test")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], label="Client_gen_test")
    plt.legend()
    plt.xlabel("Global Rounds")
    plt.ylim(YLim, 1.02)
    plt.grid()
    plt.title("AVG Clients Model (Spec-Gen) Testing Accuracy")
    plt.savefig(PLOT_PATH + alg_name+"AVGC_Spec_Gen_Testing.pdf")

    plt.figure(7)
    plt.clf()
    plt.plot(f_data['root_test'], linestyle="--", label="root test")
    plt.plot(f_data['cs_data_test'])
    plt.legend()
    plt.xlabel("Global Rounds")
    plt.ylim(0, 1.02)
    plt.grid()
    plt.title("Testing Client Specialization")
    plt.savefig(PLOT_PATH + alg_name + "C_Spec_Testing.pdf")

    plt.figure(8)
    plt.clf()
    plt.plot(f_data['root_train'], linestyle="--", label="root train")
    plt.plot(f_data['cs_data_train'])
    plt.legend()
    plt.xlabel("Global Rounds")
    plt.ylim(0, 1.02)
    plt.grid()
    plt.title("Training Client Specialization")
    plt.savefig(PLOT_PATH + alg_name + "C_Spec_Training.pdf")

    plt.figure(9)
    plt.clf()
    plt.plot(f_data['cg_data_test'])
    plt.plot(f_data['root_test'], linestyle="--", label="root test")
    plt.legend()
    plt.xlabel("Global Rounds")
    plt.ylim(0, 1.02)
    plt.grid()
    plt.title("Testing Client Generalization")
    plt.savefig(PLOT_PATH + alg_name + "C_Gen_Testing.pdf")

    plt.figure(10)
    plt.clf()
    plt.plot(f_data['cg_data_train'])
    plt.plot(f_data['root_train'], linestyle="--", label="root train")
    plt.legend()
    plt.xlabel("Global Rounds")
    plt.ylim(0, 1.02)
    plt.grid()
    plt.title("Training Client Generalization")
    plt.savefig(PLOT_PATH + alg_name + "C_Gen_Training.pdf")

    plt.show()


    print("** Summary Results: ---- Training ----")
    print("AVG Clients Specialization - Training:", f_data['cs_avg_data_train'])
    print("AVG Clients Generalization - Training::", f_data['cg_avg_data_train'])
    print("Root performance - Training:", f_data['root_train'])
    print("** Summary Results: ---- Testing ----")
    print("AVG Clients Specialization - Testing:", f_data['cs_avg_data_test'])
    print("AVG Clients Generalization - Testing:", f_data['cg_avg_data_test'])
    print("Root performance - Testing:", f_data['root_test'])

def plot_3D():
    file_name = "../results/{}_iter_{}_k_{}_w.h5".format(RUNNING_ALG, NUM_GLOBAL_ITERS, K_Levels)
    f_data = read_data(file_name)
    data = np.array(f_data['g_level_test'])
    lx = len(data[0])
    print(lx)
    # Work out matrix dimensions
    ly = len(data[:, 0])
    print(ly)
    column_names = np.arange(lx)
    row_names = np.arange(ly)
    fig = plt.figure()
    ax = Axes3D(fig)
    xpos = np.arange(0, lx, 1)  # Set up a mesh of positions
    ypos = np.arange(0, ly, 1)
    xpos, ypos = np.meshgrid(xpos + 0.25, ypos + 0.25)
    xpos = xpos.flatten()  # Convert positions to 1D array
    ypos = ypos.flatten()
    zpos = np.zeros(lx * ly)
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = data.flatten()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
    plt.show()



def plot_dem_vs_fed():
    plt.rcParams.update({'font.size': 16})
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    f_data = read_data(RS_PATH + name['avg3w'])
    print("DemLearn Global 12 iters:", f_data['root_test'][12])
    print("DemLearn Global:", f_data['root_test'][XLim-1])
    print("DemLearn C-GEN:",f_data['cg_avg_data_test'][XLim-1])

    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax1.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title("DemLearn")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2
    f_data = read_data(RS_PATH + name['prox3w'])
    print("DemLearn-P Global 12 iters:", f_data['root_test'][12])
    print("DemLearn-P C-SPE:", f_data['cs_avg_data_test'][XLim-1])
    ax2.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    ax2.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"], marker=marker["ggen"], markevery=markers_on)
    ax2.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"], marker=marker["gspe"], markevery=markers_on)
    ax2.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"], marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax2.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"], marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")

    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 1)
    ax2.set_title("DemLearn-P")
    ax2.set_xlabel("#Global Rounds")
    ax2.grid()

    # end-subfig2----begin-subfig3

    fed_data2 = read_data(RS_PATH + name['fedprox'])
    ax3.plot(fed_data2['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    ax3.plot(fed_data2['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    ax3.plot(fed_data2['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # ax3.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 1)
    ax3.grid()
    ax3.set_title("FedProx")
    ax3.set_xlabel("#Global Rounds")

    # END-subfig3-begin-subfig4

    fed_data = read_data(RS_PATH + name['fedavg'])
    print("FedAvg Global:", fed_data['root_test'][XLim-1])
    ax4.plot(fed_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    ax4.plot(fed_data['cs_avg_data_test'], label="C-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    ax4.plot(fed_data['cg_avg_data_test'], label="C-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    ax4.set_xlim(0, XLim)
    ax4.set_ylim(YLim, 1)
    ax4.grid()
    ax4.set_title("FedAvg")
    ax4.set_xlabel("#Global Rounds")
    plt.tight_layout()
    # plt.grid(linewidth=0.25)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
               prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH + "dem_vs_fed" + OUT_TYPE)
    return 0

def plot_demlearn_vs_demlearn_p():
    plt.rcParams.update({'font.size': 16})
    fig, (ax3, ax2, ax6, ax5) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    f_data = read_data(RS_PATH + name['avg3wf'])
    ax2.plot(f_data['root_test'], label="GEN", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax2.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax2.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax2.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")

    ax2.set_xlim(0, XLim)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Testing Accuracy")
    # ax2.set_title("DemLearn: $K=4$, Fixed")
    ax2.set_title("DemLearn: Fixed")
    ax2.set_xlabel("#Global Rounds")
    ax2.grid()

    f_data = read_data(RS_PATH + name['avg3w'])
    ax3.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax3.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax3.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax3.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax3.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")

    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 1)
    ax3.set_title("DemLearn")
    ax3.set_xlabel("#Global Rounds")
    ax3.grid()

    f_data = read_data(RS_PATH + name['prox3wf'])
    ax5.plot(f_data['root_test'], label="GEN", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax5.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax5.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax5.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax5.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")

    ax5.set_xlim(0, XLim)
    ax5.set_ylim(0, 1)
    # ax5.set_title("DemLearn-P: $K=4$, Fixed")
    ax5.set_xlabel("#Global Rounds")
    ax5.set_title("DemLearn-P: Fixed")
    ax5.set_xlabel("#Global Rounds")
    ax5.grid()
    f_data = read_data(RS_PATH + name['prox3w'])
    ax6.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax6.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax6.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax6.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax6.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")

    ax6.set_xlim(0, XLim)
    ax6.set_ylim(YLim, 1)
    # ax6.set_title("DemLearn-P: $K=4$")
    ax6.set_title("DemLearn-P")
    ax6.set_xlabel("#Global Rounds")
    ax6.grid()


    plt.tight_layout()
    # plt.grid(linewidth=0.25)

    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1,  ncol=5, prop={'size': 14})  # mode="expand",mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(PLOT_PATH + "dem_vs_K_vary"+OUT_TYPE)
    return 0

def plot_demlearn_p_mu_vari():
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2, ax4, ax3) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15.0, 4.4))
    f_data = read_data(RS_PATH + name['prox3wmu005'])
    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax1.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    # ax1.legend(loc="best", prop={'size': 8})
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title("DemLearn-P: $\mu=0.005$")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2
    f_data = read_data(RS_PATH + name['prox3wmu002'])

    ax2.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax2.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax2.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax2.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")

    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 1)
    ax2.set_title("DemLearn-P: $\mu=0.002$")
    ax2.set_xlabel("#Global Rounds")
    ax2.grid()

    f_data = read_data(RS_PATH + name['prox3wmu0005'])
    ax3.plot(f_data['root_test'], label="Generalization", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax3.plot(f_data['gs_level_test'][-2, :, 0], label="Group-Generalization", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax3.plot(f_data['gg_level_test'][-2, :, 0], label="Group-Specialization", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax3.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="Client-Specialization")
    ax3.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="Client-Generalization")
    # ax1.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 1)
    ax3.set_title("DemLearn-P: $\mu=0.0005$")
    ax3.set_xlabel("#Global Rounds")
    #ax3.set_ylabel("Testing Accuracy")
    ax3.grid()
    # subfig1-end---begin---subfig 2
    f_data = read_data(RS_PATH + name['prox3wmu001'])

    ax4.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax4.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax4.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax4.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax4.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")

    ax4.set_xlim(0, XLim)
    ax4.set_ylim(YLim, 1)
    ax4.set_title("DemLearn-P: $\mu=0.001$")
    ax4.set_xlabel("#Global Rounds")
    ax4.grid()
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5, prop={'size': 15})
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    # plt.grid(linewidth=0.25)

    plt.savefig(PLOT_PATH + "dem_prox_mu_vary"+OUT_TYPE)
    return 0


def plot_demlearn_gamma_vari():
    plt.rcParams.update({'font.size': 14})
    fig, (ax3, ax2, ax1) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10.0, 4.2))
    # fig, (ax3, ax2, ax1, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(13.0, 4.2))
    f_data = read_data(RS_PATH + name['avg3w'])
    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax1.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    # ax1.legend(loc="best", prop={'size': 8})
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title("DemLearn: $\gamma=0.6$")
    ax1.set_xlabel("#Global Rounds")
    # ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2
    f_data = read_data(RS_PATH + name['avg3wg08'])

    ax2.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax2.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax2.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax2.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax2.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")

    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 1)
    ax2.set_title("DemLearn: $\gamma=0.8$")
    ax2.set_xlabel("#Global Rounds")
    ax2.grid()

    f_data = read_data(RS_PATH + name['avg3g1'])
    ax3.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax3.plot(f_data['gs_level_test'][-2, :, 0], label="Group-Generalization", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax3.plot(f_data['gg_level_test'][-2, :, 0], label="Group-Specialization", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax3.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="Client-Specialization")
    ax3.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="Client-Generalization")
    # ax1.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 1)
    ax3.set_title("DemLearn: $\gamma=1.0$")
    ax3.set_xlabel("#Global Rounds")
    ax3.set_ylabel("Testing Accuracy")
    ax3.grid()

    plt.tight_layout()
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
               prop={'size': 14})  # mode="expand",
    plt.subplots_adjust(bottom=0.24)
    plt.savefig(PLOT_PATH+"dem_avg_gamma_vary"+OUT_TYPE)
    return 0


def plot_demlearn_gamma_vari_clients():
    plt.rcParams.update({'font.size': 14})
    fig, (ax3, ax2, ax1) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10.0, 3.96))
    # fig, (ax3, ax2, ax1, ax4) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(13.0, 4.2))
    f_data = read_data(RS_PATH + name['avg3w'])
    ax1.plot(f_data['cs_data_test'], linewidth=1.2)
    ax1.plot(f_data['root_test'], linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)

    # ax1.legend(loc="best", prop={'size': 8})
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(0.6, 1.01)
    ax1.set_title("Client-Spec: $\gamma=0.6$")
    ax1.set_xlabel("#Global Rounds")
    # ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2
    f_data = read_data(RS_PATH + name['avg3wg08'])
    ax2.plot(f_data['cs_data_test'], linewidth=1.2)
    ax2.plot(f_data['root_test'], linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)

    ax2.set_xlim(0, XLim)
    ax2.set_ylim(0.6, 1.01)
    ax2.set_title("Client-Spec: $\gamma=0.8$")
    ax2.set_xlabel("#Global Rounds")
    ax2.grid()

    f_data = read_data(RS_PATH + name['avg3g1'])
    ax3.plot(f_data['cs_data_test'], linewidth=1.2)
    ax3.plot(f_data['root_test'], linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)

    # ax1.legend(loc="best", prop={'size': 8})
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(0.6, 1.01)
    ax3.set_title("Client-Spec: $\gamma=1.0$")
    ax3.set_xlabel("#Global Rounds")
    ax3.set_ylabel("Testing Accuracy")
    ax3.grid()


    plt.tight_layout()
    plt.savefig(PLOT_PATH+"dem_avg_gamma_vary_clients"+OUT_TYPE)
    return 0

def plot_demlearn_w_vs_g():
    plt.rcParams.update({'font.size': 12})
    # plt.grid(linewidth=0.25)
    # fig, ((ax1, ax2, ax3),(ax4, ax5, ax6))= plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(10.0, 7))
    fig, (ax1, ax4, ax2, ax3) = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(13.0, 4.2))
    f_data = read_data(RS_PATH + name['avg3w'])
    ax1.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax1.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax1.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax1.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax1.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")
    # ax1.legend(loc="best", prop={'size': 8})
    ax1.set_xlim(0, XLim)
    ax1.set_ylim(YLim, 1)
    ax1.set_title("DemLearn:W-Clustering")
    ax1.set_xlabel("#Global Rounds")
    ax1.set_ylabel("Testing Accuracy")
    ax1.grid()
    # subfig1-end---begin---subfig 2

    ax2.plot(f_data['cg_data_test'], linewidth=0.7)
    ax2.plot(f_data['root_test'], linestyle="--", color=color["gen"], marker=marker["gen"], linewidth=2,
             markevery=markers_on)
    ax2.set_xlabel("#Global Rounds")
    ax2.grid()
    ax2.set_xlim(0, XLim)
    ax2.set_ylim(YLim, 1)
    ax2.set_title("Client-GEN: W-Clustering")

    f_data = read_data(RS_PATH + name['avg3g'])
    ax4.plot(f_data['root_test'], label="Global", linestyle="--", color=color["gen"], marker=marker["gen"],
             markevery=markers_on)
    ax4.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle="-.", color=color["ggen"],
             marker=marker["ggen"], markevery=markers_on)
    ax4.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"],
             marker=marker["gspe"], markevery=markers_on)
    ax4.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], color=color["cspe"],
             marker=marker["cspe"], markevery=markers_on,
             label="C-SPE")
    ax4.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], color=color["cgen"],
             marker=marker["cgen"], markevery=markers_on,
             label="C-GEN")

    ax4.set_xlim(0, XLim)
    ax4.set_ylim(YLim, 1)
    ax4.set_title("DemLearn: G-Clustering")
    ax4.set_xlabel("#Global Rounds")
    # ax4.set_ylabel("Testing Accuracy")
    ax4.grid()

    ax3.plot(f_data['cg_data_test'], linewidth=0.7)
    ax3.plot(f_data['root_test'], linestyle="--", color=color["gen"], marker=marker["gen"], linewidth=2,
             markevery=markers_on)
    ax3.set_xlabel("#Global Rounds")
    ax3.grid()
    ax3.set_xlim(0, XLim)
    ax3.set_ylim(YLim, 1)
    ax3.set_title("Client-GEN: G-Clustering")


    plt.tight_layout()


    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1,  ncol=5, prop={'size': 14})  # mode="expand",mode="expand", frameon=False,

    plt.subplots_adjust(bottom=0.24)
    plt.savefig(PLOT_PATH + "w_vs_g"+OUT_TYPE)
    return 0

def get_data_from_file(file_name=""):
    rs = {}
    if not file_name:
        print("File is not existing please make sure file name is correct")
    else:
        f_data = read_data(file_name)
        if('dem' in file_name):
            return  f_data['root_test'], f_data['gs_level_test'][-2,:,0], f_data['gg_level_test'][-2,:,0], f_data['cs_avg_data_test'], f_data['cg_avg_data_test']
        else:
            return f_data['root_test'], f_data['cs_avg_data_test'], f_data['cg_avg_data_test']

if __name__=='__main__':
    PLOT_PATH = "../figs/50users/f_"
    # RS_PATH =  "../results/50users/100iters/"
    RS_PATH = "../results/50users/100iters/fmnist/"
    # plot_dem_vs_fed() #plot comparision FED vs DEM
    # plot_demlearn_vs_demlearn_p() # DEM, PROX vs K level
    # plot_demlearn_p_mu_vari() # DEM Prox vs mu vary

    # ### SUPPLEMENTAL FIGS ####
    # plot_demlearn_gamma_vari() # DEM AVG vs Gamma vary
    # plot_demlearn_gamma_vari_clients()
    plot_demlearn_w_vs_g()
    # # ##-------DENDOGRAM PLOT --------------------------------------------------------------------------------#
    CLUSTER_METHOD = "gradient"
    # plot_dendo_data_dem(file_name="prox3g")  # change file_name in order to get correct file to plot   #|
    plot_dendo_data_dem(file_name="avg3g")  # change file_name in order to get correct file to plot   #|
    CLUSTER_METHOD = "weight"
    # plot_dendo_data_dem(file_name="prox3w")
    plot_dendo_data_dem(file_name="avg3w")
    # plot_dendo_data_dem(file_name=name["avg3wg08"])
    ##-------DENDOGRAM PLOT --------------------------------------------------------------------------------#
    # plot_from_file()
    plt.show()
    # dendo_data
    # dendo_data_round
    # tmp_data = read_data(RS_PATH+name["avg3w"])
    # print(tmp_data["dendo_data"].shape)
    # print(tmp_data["dendo_data_round"])

    #plot_dendrogram(tmp_data["dendo_data"],tmp_data["dendo_data_round"], "demlearn" )
