import h5py as hf
import os
import numpy as np
from clustering.Setting import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram
import matplotlib.gridspec as gridspec
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
    return   dic_data

def plot_dendrogram(rs_linkage_matrix, round, alg):
    # Plot the corresponding dendrogram
    plt.figure(1)
    plt.clf()
    # change p value to 5 if we want to get 5 levels
    plt.title('Hierarchical Clustering Dendrogram')
    rs_dendrogram = dendrogram(rs_linkage_matrix, truncate_mode='level', p=K_Levels)

    # print(rs_dendrogram['ivl'])  # x_axis of dendrogram => index of nodes or (Number of points in clusters (i))
    # print(rs_dendrogram['leaves'])  # merge points
    plt.xlabel("index of node or (Number of leaves in each cluster).")
    if(MODEL_TYPE == "cnn"):
        if(CLUSTER_METHOD == "gradient"):
            plt.ylim(0, 1.2)
        else:
            plt.ylim(0, 0.4)
    else:
        plt.ylim(0,1.5)
    plt.savefig(PLOT_PATH + alg + "_T"+str(round)+".pdf")

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
        # print(dd_data)
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

    # plt.figure(3)
    # plt.clf()
    # plt.plot(f_data['root_train'], label="Root_train", linestyle="--")
    # plt.plot(f_data['root_test'], label="Root_test", linestyle="--")
    # #add group data
    # plt.plot(np.arange(len(f_data['cs_avg_data_train'])), f_data['cs_avg_data_train'], linestyle="-",
    #          label="Client_spec_train")
    # plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], linestyle="-", label="Client_spec_test")
    # plt.plot(np.arange(len(f_data['cg_avg_data_train'])), f_data['cg_avg_data_train'], linestyle="-",
    #          label="Client_gen_train")
    # plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], linestyle="-", label="Client_gen_test")
    # plt.legend()
    # plt.xlabel("Global Rounds")
    # plt.ylim(0, 1.02)
    # plt.grid()
    # plt.title("AVG Clients Model (Spec-Gen) Accuracy")
    # plt.savefig(PLOT_PATH + alg_name + "AVGC_Spec_Gen.pdf")

    # plt.figure(3)
    # plt.clf()
    # plt.plot(f_data['root_train'], label="Root_train", linestyle="--")
    # plt.plot(np.arange(len(f_data['cs_avg_data_train'])), f_data['cs_avg_data_train'], label="Client_spec_train")
    # plt.plot(np.arange(len(f_data['cg_avg_data_train'])), f_data['cg_avg_data_train'], label="Client_gen_train")
    # plt.legend()
    # plt.xlabel("Global Rounds")
    # plt.ylim(0, 1.02)
    # plt.grid()
    # plt.title("AVG Clients Model (Spec-Gen) Training Accuracy")
    # plt.savefig(PLOT_PATH + alg_name+"AVGC_Spec_Gen_Training.pdf")

    plt.figure(4)
    plt.clf()
    plt.plot(f_data['root_test'], label="Root_test", linestyle="--")
    if("dem" in RUNNING_ALG):
        # for k in range (K_Levels):
            plt.plot(f_data['gs_level_test'][-2,:,0], label="Gr(K)_spec_test", linestyle=":")
            plt.plot(f_data['gg_level_test'][-2,:,0], label="Gr(K)_gen_test", linestyle=":")
        # plt.plot(f_data['gks_level_test'][0,:], label="Gr1(K)_spec_test", linestyle=":")
        # plt.plot(f_data['gkg_level_test'][0,:], label="Gr1(K)_gen_test", linestyle=":")
        # plt.plot(f_data['gks_level_test'][1,:], label="Gr2(K)_spec_test", linestyle=":")
        # plt.plot(f_data['gkg_level_test'][1,:], label="Gr2(K)_gen_test", linestyle=":")

    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], label="Client_spec_test")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], label="Client_gen_test")
    plt.legend()
    plt.xlabel("Global Rounds")
    plt.ylim(0, 1.02)
    plt.grid()
    plt.title("AVG Clients Model (Spec-Gen) Testing Accuracy")
    plt.savefig(PLOT_PATH + alg_name+"AVGC_Spec_Gen_Testing.pdf")

    # plt.figure(5)
    # plt.clf()
    # plt.plot(np.arange(len(self.gs_data_train)), self.gs_data_train, label="s_train")
    # plt.plot(np.arange(len(self.gs_data_test)), self.gs_data_test, label="s_test")
    # # print(self.gs_data_test)

    # plt.legend()
    # plt.grid()
    # plt.title("AVG Group Specialization")

    # plt.figure(6)
    # plt.clf()
    # plt.plot(np.arange(len(self.gg_data_train)), self.gg_data_train, label="g_train")
    # plt.plot(np.arange(len(self.gg_data_test)), self.gg_data_test, label="g_test")
    # plt.legend()
    # plt.grid()
    # plt.title("AVG Group Generalization")

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

    # cs = ['r', 'g', 'b', 'y', 'c'] * ly

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz)

    # sh()
    # ax.w_xaxis.set_ticklabels(column_names)
    # ax.w_yaxis.set_ticklabels(row_names)

    plt.show()



def get_ploting_data(RS_PATH="../results/100iters/" ):
    # list_file = os.listdir(RS_PATH)
    # list_file_name = sorted(list_file)
    # for file in list_file_name:
    #     print("\" \": \"{}\",".format(file))
    #listing file name in directory: transform into dictionary (copy from console and paste here xD)
    RS_PATH = "../results/100iters/"
    name = {
        "avg1w": "demavg_iter_100_k_1_w.h5",
        "avg2w": "demavg_iter_100_k_2_w.h5",
        "avg3g": "demavg_iter_100_k_3_g.h5",
        "avg3w": "demavg_iter_100_k_3_w.h5",
        "prox1w": "demprox_iter_100_k_1_w.h5",
        "prox2w": "demprox_iter_100_k_2_w.h5",
        "prox3w": "demprox_iter_100_k_3_w.h5",
        "fedavg": "fedavg_iter_100.h5",
        "fedprox": "fedprox_iter_100.h5",
        "avg3b08": "demavg_iter_100_k_3_w_beta_0_8.h5",
        "avg3wdecay": "demavg_iter_100_k_3_w_decay.h5",
        "avg3wg08": "demavg_iter_100_k_3_w_gamma_0_8.h5",
        "avg3g1": "demavg_iter_100_k_3_w_gamma_1.h5",
        "prox3wg08": "demprox_iter_100_k_3_w_gamma_0_8.h5",
        "prox3wg1": "demprox_iter_100_k_3_w_gamma_1.h5",
        "prox3wmu0001": "demprox_iter_100_k_3_w_mu_0001.h5",
        "prox3wmu0005": "demprox_iter_100_k_3_w_mu_0005.h5",
        "prox3wmu005": "demprox_iter_100_k_3_w_mu_005.h5"
    }
    markers_on = [ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90] #np.arange(0,100,10)


    plt.figure(1)
    plt.subplot(131)
    f_data = read_data(RS_PATH + name['prox1w'])
    plt.plot(f_data['root_test'], label="GEN", linestyle="--", marker='8',markevery=markers_on)
    # plt.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle=":",marker='s',markevery=markers_on)
    # plt.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", marker='P',markevery=markers_on)
    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], marker='p',markevery=markers_on,label="C-SPE")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], marker='*',markevery=markers_on,label="C-GEN")
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0,1)
    plt.title("DEM-LEARN")
    plt.xlabel("#Round \n a)")
    plt.ylabel("Accuracy")
    plt.grid()
    # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3, prop={'size': 6})
    # plt.tight_layout()
    # plt.legend()

    # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3,prop={'size': 6})
    # plt.tight_layout()
    plt.subplot(133)
    fed_data2 = read_data(RS_PATH + name['fedprox'])
    plt.plot(fed_data2['root_test'], label="GEN", linestyle="-.", marker='8', markevery=markers_on)
    plt.plot(fed_data2['cs_avg_data_test'], label="C-SPE", marker='p', markevery=markers_on)
    plt.plot(fed_data2['cg_avg_data_test'], label="C-GEN", marker='*', markevery=markers_on)
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.grid()
    plt.title("FED-PROX")
    plt.xlabel("#Round \n c)")
    # plt.ylabel("Accuracy")
    # plt.tight_layout()
    # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=5, prop={'size': 6})
    plt.subplot(132)
    fed_data = read_data(RS_PATH+name['fedavg'])
    plt.plot(fed_data['root_test'], label="GEN", linestyle="-.", marker='8', markevery=markers_on)
    plt.plot(fed_data['cs_avg_data_test'], label="C-SPE", marker='p', markevery=markers_on)
    plt.plot(fed_data['cg_avg_data_test'], label="C-GEN", marker='*', markevery=markers_on)
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.grid()
    plt.title("FED-AVG")
    plt.xlabel("#Round \n d)")
    # plt.ylabel("Accuracy")
    # plt.show()
    #---------------------//K=1, K=2, K=3//---------------------------------#
    plt.figure(2)
    plt.subplot(231)
    f_data = read_data(RS_PATH + name['avg1w'])
    plt.plot(f_data['root_test'], label="GEN", linestyle="--", marker='8', markevery=markers_on)
    plt.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle=":", marker='s', markevery=markers_on)
    plt.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", marker='P', markevery=markers_on)
    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], marker='p', markevery=markers_on,
             label="C-SPE")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], marker='*', markevery=markers_on,
             label="C-GEN")
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.title("DEM-AVG K =1")
    plt.xlabel("#Round \n a)")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.subplot(232)
    f_data = read_data(RS_PATH + name['avg2w'])
    plt.plot(f_data['root_test'], label="GEN", linestyle="--", marker='8', markevery=markers_on)
    plt.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle=":", marker='s', markevery=markers_on)
    plt.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", marker='P', markevery=markers_on)
    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], marker='p', markevery=markers_on,
             label="C-SPE")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], marker='*', markevery=markers_on,
             label="C-GEN")
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.title("DEM-AVG K =2")
    plt.xlabel("#Round \n a)")
    plt.ylabel("Accuracy")
    plt.grid()
    # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3, prop={'size': 6})
    # plt.tight_layout()
    # plt.legend()
    plt.subplot(233)
    f_data = read_data(RS_PATH + name['avg3w'])
    plt.plot(f_data['root_test'], label="GEN", linestyle="--", marker='8', markevery=markers_on)
    plt.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle=":", marker='s', markevery=markers_on)
    plt.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", marker='P', markevery=markers_on)
    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], marker='p', markevery=markers_on,
             label="C-SPE")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], marker='*', markevery=markers_on,
             label="C-GEN")
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.title("DEM-AVG K=3")
    plt.xlabel("#Round \n c)")
    # plt.ylabel("Accuracy")
    plt.grid()
    # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3, prop={'size': 6})
    # plt.tight_layout()
    # plt.legend()
    plt.subplot(234)
    f_data = read_data(RS_PATH + name['prox1w'])
    plt.plot(f_data['root_test'], label="GEN", linestyle="--", marker='8', markevery=markers_on)
    plt.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle=":", marker='s', markevery=markers_on)
    plt.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", marker='P', markevery=markers_on)
    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], marker='p', markevery=markers_on,
             label="C-SPE")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], marker='*', markevery=markers_on,
             label="C-GEN")
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.title("DEM-PROX K=1")
    plt.xlabel("#Round \n b)")
    # plt.ylabel("Accuracy")
    plt.grid()
    plt.subplot(235)
    f_data = read_data(RS_PATH + name['prox2w'])
    plt.plot(f_data['root_test'], label="GEN", linestyle="--", marker='8', markevery=markers_on)
    plt.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle=":", marker='s', markevery=markers_on)
    plt.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", marker='P', markevery=markers_on)
    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], marker='p', markevery=markers_on,
             label="C-SPE")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], marker='*', markevery=markers_on,
             label="C-GEN")
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.title("DEM-PROX K=2")
    plt.xlabel("#Round \n b)")
    # plt.ylabel("Accuracy")
    plt.grid()
    # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3,prop={'size': 6})
    # plt.tight_layout()

    plt.subplot(236)
    f_data = read_data(RS_PATH + name['prox3w'])
    plt.plot(f_data['root_test'], label="GEN", linestyle="--", marker='8', markevery=markers_on)
    plt.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle=":", marker='s', markevery=markers_on)
    plt.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", marker='P', markevery=markers_on)
    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], marker='p', markevery=markers_on,
             label="C-SPE")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], marker='*', markevery=markers_on,
             label="C-GEN")
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.title("DEM-PROX K = 3")
    plt.xlabel("#Round \n d)")
    # plt.ylabel("Accuracy")
    plt.grid()

    #--------------------//-----gamma-------//--------------//
    plt.figure(3)
    plt.subplot(141)
    f_data = read_data(RS_PATH + name['avg3w'])
    plt.plot(f_data['root_test'], label="GEN", linestyle="--", marker='8', markevery=markers_on)
    plt.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle=":", marker='s', markevery=markers_on)
    plt.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", marker='P', markevery=markers_on)
    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], marker='p', markevery=markers_on,
             label="C-SPE")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], marker='*', markevery=markers_on,
             label="C-GEN")
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.title("DEM-AVG $\gamma = 0.6$")
    plt.xlabel("#Round \n a)")
    plt.ylabel("Accuracy")
    plt.grid()
    # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3, prop={'size': 6})
    # plt.tight_layout()
    # plt.legend()
    plt.subplot(142)
    f_data = read_data(RS_PATH + name['avg3wdecay'])
    plt.plot(f_data['root_test'], label="GEN", linestyle="--", marker='8', markevery=markers_on)
    plt.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle=":", marker='s', markevery=markers_on)
    plt.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", marker='P', markevery=markers_on)
    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], marker='p', markevery=markers_on,
             label="C-SPE")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], marker='*', markevery=markers_on,
             label="C-GEN")
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.title("DEM-AVG $\gamma$ decay")
    plt.xlabel("#Round \n b)")
    # plt.ylabel("Accuracy")
    plt.grid()
    # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3,prop={'size': 6})
    # plt.tight_layout()
    plt.subplot(143)
    f_data = read_data(RS_PATH + name['avg3wg08'])
    plt.plot(f_data['root_test'], label="GEN", linestyle="--", marker='8', markevery=markers_on)
    plt.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle=":", marker='s', markevery=markers_on)
    plt.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", marker='P', markevery=markers_on)
    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], marker='p', markevery=markers_on,
             label="C-SPE")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], marker='*', markevery=markers_on,
             label="C-GEN")
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.title("DEM-AVG $\gamma = 0.8$")
    plt.xlabel("#Round \n c)")
    # plt.ylabel("Accuracy")
    plt.grid()
    # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3, prop={'size': 6})
    # plt.tight_layout()
    # plt.legend()
    plt.subplot(144)
    f_data = read_data(RS_PATH + name['avg3g1'])
    plt.plot(f_data['root_test'], label="GEN", linestyle="--", marker='8', markevery=markers_on)
    plt.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle=":", marker='s', markevery=markers_on)
    plt.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", marker='P', markevery=markers_on)
    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], marker='p', markevery=markers_on,
             label="C-SPE")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], marker='*', markevery=markers_on,
             label="C-GEN")
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.title("DEM-AVG $\gamma = 1.0$")
    plt.xlabel("#Round\n d)")
    # plt.ylabel("Accuracy")
    plt.grid()

    #-----------PROX--------------

    plt.figure(4)
    plt.subplot(131)
    f_data = read_data(RS_PATH + name['prox3w'])
    plt.plot(f_data['root_test'], label="GEN", linestyle="--", marker='8', markevery=markers_on)
    plt.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle=":", marker='s', markevery=markers_on)
    plt.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", marker='P', markevery=markers_on)
    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], marker='p', markevery=markers_on,
             label="C-SPE")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], marker='*', markevery=markers_on,
             label="C-GEN")
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.title("DEM-PROX $\gamma = 0.6$")
    plt.xlabel("#Round \n a)")
    plt.ylabel("Accuracy")
    plt.grid()
    # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3, prop={'size': 6})
    # plt.tight_layout()
    # plt.legend()
    plt.subplot(132)
    f_data = read_data(RS_PATH + name['prox3wg08'])
    plt.plot(f_data['root_test'], label="GEN", linestyle="--", marker='8', markevery=markers_on)
    plt.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle=":", marker='s', markevery=markers_on)
    plt.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", marker='P', markevery=markers_on)
    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], marker='p', markevery=markers_on,
             label="C-SPE")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], marker='*', markevery=markers_on,
             label="C-GEN")
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.title("DEM-PROX $\gamma=0.8$")
    plt.xlabel("#Round \n b)")
    # plt.ylabel("Accuracy")
    plt.grid()
    # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3,prop={'size': 6})
    # plt.tight_layout()
    plt.subplot(133)
    f_data = read_data(RS_PATH + name['prox3wg1'])
    plt.plot(f_data['root_test'], label="GEN", linestyle="--", marker='8', markevery=markers_on)
    plt.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle=":", marker='s', markevery=markers_on)
    plt.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", marker='P', markevery=markers_on)
    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], marker='p', markevery=markers_on,
             label="C-SPE")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], marker='*', markevery=markers_on,
             label="C-GEN")
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.title("DEM-PROX $\gamma = 1.0$")
    plt.xlabel("#Round \n c)")
    # plt.ylabel("Accuracy")
    plt.grid()
    # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3, prop={'size': 6})
    # plt.tight_layout()
    # plt.legend()
    #================MU================//
    plt.figure(5)
    plt.subplot(131)
    f_data = read_data(RS_PATH + name['prox3wmu0001'])
    plt.plot(f_data['root_test'], label="GEN", linestyle="--", marker='8', markevery=markers_on)
    plt.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle=":", marker='s', markevery=markers_on)
    plt.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", marker='P', markevery=markers_on)
    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], marker='p', markevery=markers_on,
             label="C-SPE")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], marker='*', markevery=markers_on,
             label="C-GEN")
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.title("DEM-PROX $\gamma = 0.6, \mu=0.0001$")
    plt.xlabel("#Round \n a)")
    plt.ylabel("Accuracy")
    plt.grid()
    # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3, prop={'size': 6})
    # plt.tight_layout()
    # plt.legend()
    plt.subplot(132)
    f_data = read_data(RS_PATH + name['prox3wmu0005'])
    plt.plot(f_data['root_test'], label="GEN", linestyle="--", marker='8', markevery=markers_on)
    plt.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle=":", marker='s', markevery=markers_on)
    plt.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", marker='P', markevery=markers_on)
    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], marker='p', markevery=markers_on,
             label="C-SPE")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], marker='*', markevery=markers_on,
             label="C-GEN")
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.title("DEM-PROX $\gamma=0.6, \mu=0.0005$")
    plt.xlabel("#Round \n b)")
    # plt.ylabel("Accuracy")
    plt.grid()
    # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3,prop={'size': 6})
    # plt.tight_layout()
    plt.subplot(133)
    f_data = read_data(RS_PATH + name['prox3wmu005'])
    plt.plot(f_data['root_test'], label="GEN", linestyle="--", marker='8', markevery=markers_on)
    plt.plot(f_data['gs_level_test'][-2, :, 0], label="G-GEN", linestyle=":", marker='s', markevery=markers_on)
    plt.plot(f_data['gg_level_test'][-2, :, 0], label="G-SPE", linestyle="-.", marker='P', markevery=markers_on)
    plt.plot(np.arange(len(f_data['cs_avg_data_test'])), f_data['cs_avg_data_test'], marker='p', markevery=markers_on,
             label="C-SPE")
    plt.plot(np.arange(len(f_data['cg_avg_data_test'])), f_data['cg_avg_data_test'], marker='*', markevery=markers_on,
             label="C-GEN")
    plt.legend(loc="best", prop={'size': 8})
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.title("DEM-PROX $\gamma = 0.6, \mu=0.05$")
    plt.xlabel("#Round \n c)")
    # plt.ylabel("Accuracy")
    plt.grid()
    # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3, prop={'size': 6})
    # plt.tight_layout()
    # plt.legend()



    plt.show()

    return  0
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
    RS_PATH =  "../results/100iters/"

    get_ploting_data()

    # markers_on = np.arange(0, 100, 10)
    # print(markers_on)
    # x = np.linspace(0, 2 * np.pi, 400)
    # y = np.sin(x ** 2)
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].plot(x, y)
    # axs[0, 0].set_title('Axis [0, 0]')
    # axs[0, 1].plot(x, y, 'tab:orange')
    # axs[0, 1].set_title('Axis [0, 1]')
    # axs[1, 0].plot(x, -y, 'tab:green', label="test")
    # #axs[1, 0].set_title('Axis [1, 0]')
    # axs[1, 1].plot(x, -y, 'tab:red')
    # axs[1, 1].set_title('Axis [1, 1]')
    # axs[1, 0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
    #             mode="expand", borderaxespad=0, ncol=3)
    # for ax in axs.flat:
    #     ax.set(xlabel='x-label', ylabel='y-label')
    #
    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()
    # #for ax in axs.flat:
    #     # ax.label_outer()
    # plt.show()