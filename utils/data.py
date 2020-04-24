import h5py as hf
import numpy as np
from clustering.Setting import *


def  write_file(file_name = "./results/untitled.h5", **kwargs):
    with hf.File(file_name, "w") as data_file:
        for key, value in kwargs.items():
            #print("%s == %s" % (key, value))
            data_file.create_dataset(key, data=value)
    print("Successfully save to file!")
def read_data(file_name = "./results/untitled.h5"):
    rs = []
    dic_data = {}
    with hf.File(file_name, "r") as f:
        # List all groups
        #print("Keys: %s" % f.keys())
        for key in f.keys():
            rs.append( [key, f[key][:]] )
            dic_data[key] = f[key][:]
    return   dic_data

if __name__=='__main__':
    # a = [[0.09540889526542325, 0.1135953840605842], [0.9921090387374462, 0.998918139199423], [0.9921090387374462, 0.999278759466282], [0.994261119081779, 1.0], [0.9928263988522238, 1.0], [0.9928263988522238, 1.0], [0.9921090387374462, 1.0], [0.9921090387374462, 1.0], [0.9921090387374462, 1.0], [0.9921090387374462, 1.0]]
    # b = np.asarray(a)
    # c = np.asarray([1,2,3,4,5,6,7,8])
    # write_file( file_name="../results/tri_test.h5", b=b, c=c)
    file_name = "../results/alg_{}_iter_{}_k_{}_w.h5".format(RUNNING_ALG, NUM_GLOBAL_ITERS, K_Levels)
    dt = read_data(file_name)
    print(dt)
    # print(dt['cs_avg_data_test'])
    # print(dt['cs_avg_data_train'])
    # print(dt['root_test'])
    # print(dt['root_train'])