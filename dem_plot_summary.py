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
YLim=0.1
#Global variable
markers_on = 10 #maker only at x = markers_on[]
OUT_TYPE = ".pdf" #.eps or .pdf #output figure type

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
    # print(dendo_data_round)
    i = 0
    for i in range(8):
        plt.subplot(num_plots[i])
        plot_dendrogram(dendo_data[i*Den_GAP], dendo_data_round[i*Den_GAP], RUNNING_ALG)
    plt.tight_layout()
    plt.savefig(PLOT_PATH+ DATASET+"_den_" + file_name+OUT_TYPE)
    return 0

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
    plt.savefig(PLOT_PATH+ DATASET + "_dem_vs_fed" + OUT_TYPE)
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
    plt.savefig(PLOT_PATH+ DATASET + "_dem_vs_K_vary"+OUT_TYPE)
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

    plt.savefig(PLOT_PATH+ DATASET + "_dem_prox_mu_vary"+OUT_TYPE)
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
    plt.savefig(PLOT_PATH+ DATASET+"_dem_avg_gamma_vary"+OUT_TYPE)
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
    plt.savefig(PLOT_PATH+ DATASET+"_dem_avg_gamma_vary_clients"+OUT_TYPE)
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
    plt.savefig(PLOT_PATH + DATASET + "_w_vs_g"+OUT_TYPE)
    return 0

def plot_all_figs():
    global CLUSTER_METHOD
    plot_dem_vs_fed()  # plot comparision FED vs DEM
    plot_demlearn_vs_demlearn_p()  # DEM, PROX vs K level
    plot_demlearn_p_mu_vari()  # DEM Prox vs mu vary

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
    ##-------DENDOGRAM PLOT --------------------------------------------------------------------------------#
    # plot_from_file()
    # plt.show()


if __name__=='__main__':
    PLOT_PATH = "."+PLOT_PATH
    RS_PATH = "."+RS_PATH

    #### PLOT MNIST #####
    print("----- PLOT MNIST ------")
    DATASET = "mnist"
    NUM_GLOBAL_ITERS = 60
    XLim = NUM_GLOBAL_ITERS
    Den_GAP = 4

    name = {
        "avg1w": "mnist_demlearn_I60_K1_T2_b1-0_dTrue_m0-002_w.h5",
        "avg1wf": "mnist_demlearn_I60_K1_T60_b1-0_dTrue_m0-002_w.h5",
        # "avg2w": "demlearn_iter_60_k_2_w.h5",
        "avg3g": "mnist_demlearn_I60_K3_T2_b1-0_dTrue_m0-002_g.h5",
        "avg3w": "mnist_demlearn_I60_K3_T2_b1-0_dTrue_m0-002_w.h5",
        "avg3wf": "mnist_demlearn_I60_K3_T60_b1-0_dTrue_m0-002_w.h5",
        "prox1w": "mnist_demlearn-p_I60_K1_T2_b1-0_dTrue_m0-002_w.h5",
        "prox1wf": "mnist_demlearn-p_I60_K1_T60_b1-0_dTrue_m0-002_w.h5",
        # "prox2w": "demlearn-p_iter_60_k_2_w.h5",
        "prox3w": "mnist_demlearn-p_I60_K3_T2_b1-0_dTrue_m0-002_w.h5",
        "prox3g": "mnist_demlearn-p_I60_K3_T2_b1-0_dTrue_m0-002_g.h5",
        "prox3wf": "mnist_demlearn-p_I60_K3_T60_b1-0_dTrue_m0-002_w.h5",
        "fedavg": "mnist_fedavg_I60.h5",
        "fedprox": "mnist_fedprox_I60.h5",
        # "avg3b08": "demlearn_iter_60_k_3_w_beta_0_8.h5",
        # "avg3wdecay": "demlearn_iter_60_k_3_w_decay.h5",
        # "avg3wg08": "demlearn_iter_60_k_3_w_gamma_0_8.h5",
        # "avg3g1": "demlearn_iter_60_k_3_w_gamma_1.h5",
        # "prox3wg08": "demlearn-p_iter_60_k_3_w_gamma_0_8.h5",
        # "prox3wg1": "demlearn-p_iter_60_k_3_w_gamma_1.h5",
        "prox3wmu001": "mnist_demlearn-p_I60_K3_T2_b1-0_dTrue_m0-001_w.h5",
        "prox3wmu002": "mnist_demlearn-p_I60_K3_T2_b1-0_dTrue_m0-002_w.h5",
        "prox3wmu0005": "mnist_demlearn-p_I60_K3_T2_b1-0_dTrue_m0-0005_w.h5",
        "prox3wmu005": "mnist_demlearn-p_I60_K3_T2_b1-0_dTrue_m0-005_w.h5"
        }

    plot_all_figs()

    #### PLOT FASHION-MNIST #####
    print("----- PLOT FASHION-MNIST ------")
    DATASET = "fmnist"
    NUM_GLOBAL_ITERS = 100
    XLim = NUM_GLOBAL_ITERS
    Den_GAP = 7

    name = {
        "avg1w": "fmnist_demlearn_I100_K1_T2_b1-0_dTrue_m0-001_w.h5",
        "avg1wf": "fmnist_demlearn_I100_K1_T100_b1-0_dTrue_m0-001_w.h5",
        # "avg2w": "demlearn_iter_100_k_2_w.h5",
        "avg3g": "fmnist_demlearn_I100_K3_T2_b1-0_dTrue_m0-001_g.h5",
        "avg3w": "fmnist_demlearn_I100_K3_T2_b1-0_dTrue_m0-001_w.h5",
        "avg3wf": "fmnist_demlearn_I100_K3_T100_b1-0_dTrue_m0-001_w.h5",
        "prox1w": "fmnist_demlearn-p_I100_K1_T2_b1-0_dTrue_m0-001_w.h5",
        "prox1wf": "fmnist_demlearn-p_I100_K1_T100_b1-0_dTrue_m0-001_wh5",
        # "prox2w": "demlearn-p_iter_100_k_2_w.h5",
        "prox3w": "fmnist_demlearn-p_I100_K3_T2_b1-0_dTrue_m0-001_w.h5",
        "prox3g": "fmnist_demlearn-p_I100_K3_T2_b1-0_dTrue_m0-001_g.h5",
        "prox3wf": "fmnist_demlearn-p_I100_K3_T100_b1-0_dTrue_m0-001_w.h5",
        "fedavg": "fmnist_fedavg_I100.h5",
        "fedprox": "fmnist_fedprox_I100.h5",
        # "avg3b08": "demlearn_iter_100_k_3_w_beta_0_8.h5",
        # "avg3wdecay": "demlearn_iter_60_k_3_w_decay.h5",
        # "avg3wg08": "demlearn_iter_100_k_3_w_gamma_0_8.h5",
        # "avg3g1": "demlearn_iter_100_k_3_w_gamma_1.h5",
        # "prox3wg08": "demlearn-p_iter_100_k_3_w_gamma_0_8.h5",
        # "prox3wg1": "demlearn-p_iter_100_k_3_w_gamma_1.h5",
        "prox3wmu001": "fmnist_demlearn-p_I100_K3_T2_b1-0_dTrue_m0-001_w.h5",
        "prox3wmu002": "fmnist_demlearn-p_I100_K3_T2_b1-0_dTrue_m0-002_w.h5",
        "prox3wmu0005": "fmnist_demlearn-p_I100_K3_T2_b1-0_dTrue_m0-0005_w.h5",
        "prox3wmu005": "fmnist_demlearn-p_I100_K3_T2_b1-0_dTrue_m0-005_w.h5"
        }

    plot_all_figs()

