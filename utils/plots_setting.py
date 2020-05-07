import matplotlib.pyplot as plt
from Setting import *

colors1 = ["dodgerblue", "mediumseagreen", "coral"]
colors2 = ["skyblue", "mediumaquamarine", "sandybrown"]
colors3 = ["dodgerblue", "mediumseagreen", "coral", "crimson", "violet"]

markers = ["o", "x", "s", "v", "d", "^", "<", ">", "+", "*", "h", "p"]
patterns = ["", "."]

fig_size = (8,6) #For 3 power
fig_size1 = (7.1,4.4)
# fig_size2 = (5.8,3.8)
fig_size3 = (6.5,6.5)  # 3 plots in vertical
fig_size4 = (18, 4.2)  # 3 UA in vertical
labelsize = 15
legendsize = labelsize - 1
alg_labels = ["CMM","DMM, $K=1$", "DMM, $K=2$"]
alg_labels1 = ["CMM", "Activate_100%", "Activate_" + str(int(f_min * 100)) + "%"]
BS_labels = ["BS0", "BS1", "BS2", "BS3", "BS4"]
Site_labels = ["Site 0", "Site 1", "Site 2", "Site 3", "Site 4"]
PLOT_PATH = "Figs/" if SIMULATION_MODE<3 else "Figs/Scale/"

def plot_convergence(f_scaling,  k):
#    colors=["b","r","k"]
    plt.figure(1,figsize=fig_size1)
    plt.clf()
    for j in range(Numb_BS):
        plt.plot(range(0,k),f_scaling[j,0:k],marker=markers[j], markersize=6,markevery=1,linestyle=":",label=Site_labels[j])

    plt.legend(loc=1,fontsize=legendsize)
    plt.xlabel("Iterations",fontsize=labelsize)
    plt.ylabel("Scaling factor $f$",fontsize=labelsize)
    # plt.show()
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    plt.savefig(PLOT_PATH+"Solution_Convergence.pdf")




