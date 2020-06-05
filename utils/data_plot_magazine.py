import h5py as hf
from Setting import *
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})
plt.rcParams['lines.linewidth'] = 2
#Global variable
# markers_on = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
markers_on = [0, 10, 20, 30, 40, 50]
RS_PATH = "../results/50users/100iters"
name = {
        # "avg1w": "demlearn_iter_100_k_1_w.h5",
        # "avg2w": "demlearn_iter_100_k_2_w.h5",
        # "avg3g": "demlearn_iter_100_k_3_g.h5",
        # "avg3w": "demlearn_iter_100_k_3_w.h5",
        "prox1w": "demlearn-p_iter_60_k_1_w.h5",
        # "prox2w": "demlearn-p_iter_100_k_2_w.h5",
        # "prox3w": "demlearn-p_iter_100_k_3_w.h5",
        "fedavg": "fedavg_iter_60.h5",
        "fedprox": "fedprox_iter_60.h5",
        # "avg3b08": "demlearn_iter_100_k_3_w_beta_0_8.h5",
        # "avg3wdecay": "demlearn_iter_100_k_3_w_decay.h5",
        # "avg3wg08": "demlearn_iter_100_k_3_w_gamma_0_8.h5",
        # "avg3g1": "demlearn_iter_100_k_3_w_gamma_1.h5",
        # "prox3wg08": "demlearn-p_iter_100_k_3_w_gamma_0_8.h5",
        # "prox3wg1": "demlearn-p_iter_100_k_3_w_gamma_1.h5",
        # "prox3wmu0001": "demlearn-p_iter_100_k_3_w_mu_0001.h5",
        # "prox3wmu0005": "demlearn-p_iter_100_k_3_w_mu_0005.h5",
        # "prox3wmu005": "demlearn-p_iter_100_k_3_w_mu_005.h5"
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
    "gen": "x",
    "gspe": "d",
    "ggen": "P",
    "cspe": ">",
    "cgen": "o"
}
PLOT_PATH = "../figs/"

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



def plot_dem_vs_fed(dataset="mnist"):

    # fig, (ax2, ax4, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(9.0, 4.2))
    fig, (ax2, ax4) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(9.0, 4.5))

    f_data = read_data(RS_PATH + name['prox1w'])

    ax2.plot(f_data['root_test'], label="Regional", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    # ax2.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle=":", color=color["ggen"], marker=marker["ggen"], markevery=markers_on)
    # ax2.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", color=color["gspe"], marker=marker["gspe"], markevery=markers_on)
    ax2.plot(f_data['cs_avg_data_test'], color=color["cspe"], marker=marker["cspe"], markevery=markers_on,
             label="Client Specialization")
    ax2.plot(f_data['cg_avg_data_test'], color=color["cgen"], marker=marker["cgen"], markevery=markers_on,
             label="Client Generalization")

    ax2.set_xlim(0, 60)
    ax2.set_ylim(0.1, 1.01)
    ax2.set_ylabel("Testing Accuracy")
    ax2.set_title("DemLearn")
    ax2.set_xlabel("#Global Rounds")
    ax2.grid()

    # # end-subfig2----begin-subfig3
    #
    # fed_data2 = read_data(RS_PATH + name['fedprox'])
    # ax3.plot(fed_data2['root_test'], label="Regional", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    # ax3.plot(fed_data2['cs_avg_data_test'], label="Client-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    # ax3.plot(fed_data2['cg_avg_data_test'], label="Client-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # # ax3.legend(loc="best", prop={'size': 8})
    # ax3.set_xlim(0, 40)
    # ax3.set_ylim(0, 1)
    # ax3.grid()
    # ax3.set_title("FedProx")
    # ax3.set_xlabel("#Global Rounds")

    # END-subfig3-begin-subfig4

    fed_data = read_data(RS_PATH + name['fedavg'])
    ax4.plot(fed_data['root_test'], label="Regional", linestyle="--", color=color["gen"], marker=marker["gen"], markevery=markers_on)
    ax4.plot(fed_data['cs_avg_data_test'], label="Client-SPE", color=color["cspe"], marker=marker["cspe"], markevery=markers_on)
    ax4.plot(fed_data['cg_avg_data_test'], label="Client-GEN", color=color["cgen"], marker=marker["cgen"], markevery=markers_on)
    # plt.legend(loc="best", prop={'size': 8})
    ax4.set_xlim(0, 60)
    ax4.set_ylim(0.1, 1.01)
    ax4.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax4.grid()
    ax4.set_title("FedAvg")
    ax4.set_xlabel("#Global Rounds")
    plt.tight_layout()
    # plt.grid(linewidth=0.25)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", borderaxespad=0.1, ncol=5,
               prop={'size': 16})  # mode="expand",  mode="expand", frameon=False,
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(PLOT_PATH + dataset + "Dem-Learn.eps")
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
    PLOT_PATH = "../figs/"
    RS_PATH =  "../results/100users/mnist/"
    plot_dem_vs_fed("mnist") #plot comparision FED vs DEM
    RS_PATH = "../results/100users/fmnist/"
    plot_dem_vs_fed("fmnist")  # plot comparision FED vs DEM
    # plot_demlearn_vs_demlearn-p() # DEM, PROX vs K level
    # plot_demlearn_gamma_vari() # DEM AVG vs Gamma vary
    # plot_demlearn-p_mu_vari() # DEM Prox vs mu vary
    #-------DENDOGRAM PLOT --------------------------------------------------------------------------------#
    #
    #plot_dendo_data_dem(file_name=name["avg3w"]) #change file_name in order to get correct file to plot   #|
    #
    # -------DENDOGRAM PLOT --------------------------------------------------------------------------------#
    plt.show()
    # dendo_data
    # dendo_data_round
    # tmp_data = read_data(RS_PATH+name["avg3w"])
    # print(tmp_data["dendo_data"].shape)
    # print(tmp_data["dendo_data_round"])

    #plot_dendrogram(tmp_data["dendo_data"],tmp_data["dendo_data_round"], "demlearn" )
