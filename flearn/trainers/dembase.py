import numpy as np
import tensorflow as tf
from tqdm import tqdm

from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad
from clustering.hierrachical_clustering import *
from clustering.Setting import *
from flearn.models.demclient import DemClient
import h5py


class DemBase(object):
    def __init__(self, params, learner, dataset):
        # transfer parameters to self

        for key, val in params.items(): setattr(self, key, val);
        self.parameters = params
        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(*params['model_params'], self.inner_opt, self.seed)
        #initilzation of clients
        self.Weight_dimension = 10

        self.clients = self.setup_clients(dataset, self.client_model)
        self.N_clients = len(self.clients)
        self.TreeRoot = None
        self.gamma = 1.   #soft or hard update in hierrachical averaging
        self.beta  = 1.
        self.Hierrchical_Method = "Weight" ### "Gradient" or "Weight"

        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = self.client_model.get_params()
        self.model_shape = (self.latest_model[0].shape,self.latest_model[1].shape) #weight, bias dimension
        print("Model Shape:", self.model_shape)
        # initialize system metrics
        self.metrics = Metrics(self.clients, params)
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc = [], [], []
        self.test_accs = np.zeros(K_Levels+1)
        self.train_accs = np.zeros(K_Levels+1)
        self.count_grs = np.zeros(K_Levels+1)
        self.g_data_test = []  # avg generalization client accuracy test 
        self.g_data_train = [] # avg generalization client accuracy train
        self.s_data_test = []   # avg specialization client test accuracy 
        self.s_data_train = [] # avg specialization client train accuracy
        self.gs_data_test = []  # specialization of group test accuracy
        self.gs_data_test.append(np.zeros(K_Levels+1 ))
        self.gg_data_test = [] # generalization of group test accuracy
        self.gs_data_train = []  # specialization of group train accuracy
        self.gs_data_train.append(np.zeros(K_Levels + 1))
        self.gg_data_train = []  # generalization of group train accuracy
        self.cs_data_test = np.zeros((self.num_rounds, self.N_clients))
        self.cs_data_train = np.zeros((self.num_rounds, self.N_clients))
        self.cg_data_test = np.zeros((self.num_rounds, self.N_clients))
        self.cg_data_train = np.zeros((self.num_rounds, self.N_clients))


    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [DemClient(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
        # print("ID of Client 1:",all_clients[0]._id)
        return all_clients

    def create_matrix(self,csolns):
        w_list =[]
        self.model_shape = (csolns[0][1][0].shape,csolns[0][1][1].shape)  #weight, bias dimension
        print("Model Shape:", self.model_shape)
        for w in csolns:
            # print("Weight:", w[1][0])
            # print("Bias:", w[1][1])
            w_list.append( np.concatenate( (w[1][0].flatten(),w[1][1]), axis=0)   )

        self.Weight_dimension = len(w_list[0])
        return w_list

    def run_clustering(self):
        p_list =[]
        for c in self.clients:
            # print("Weight:", w[1][0])
            # print("Bias:", w[1][1])
            if(self.Hierrchical_Method == "Weight"):
                p_list.append( np.concatenate( (c.gmodel[0].flatten(),c.gmodel[1]), axis=0)   )
            else:
                p_list.append(np.concatenate((c.grad[0].flatten(), c.grad[1]), axis=0))

        self.Weight_dimension = len(p_list[0])
        if (self.Hierrchical_Method == "Weight"):
            return weight_clustering(p_list)
        else:
            return gradient_clustering(p_list)

    # def create_g_matrix(self,cgrads):
    #     g_list =[]
    #     self.model_shape = (cgrads[0][0].shape,cgrads[0][1].shape)  #weight, bias dimension
    #     print("Model Shape:", self.model_shape)
    #     for g in cgrads:
    #         print("Grad Weight Shape:", g[0].shape)
    #         print("Grad Bias Shape:", g[1].shape)
    #         g_list.append( np.concatenate( (g[0].flatten(),g[1]), axis=0)   )
    #
    #     self.Weight_dimension = len(g_list[0])
    #     return g_list


    def update_generalized_model(self,node,mode="hard"):
        # print("Node id:", node._id, node._type)
        childs = node.childs
        if childs:
            # node.numb_clients = node.count_clients()
            node.in_clients = node.collect_clients()
            node.numb_clients = len(node.in_clients)
            # print(node.numb_clients)
            # print(self.Weight_dimension)
            rs_w = np.zeros(self.model_shape[0])
            rs_b = np.zeros(self.model_shape[1])
            for child in childs:
                gmd = self.update_generalized_model(child,mode)
                # print("shape=",gmd.shape)
                rs_w += child.numb_clients * gmd[0].astype(np.float64)  #weight
                rs_b += child.numb_clients * gmd[1].astype(np.float64)  #bias
            avg_w = 1.0 * rs_w /node.numb_clients
            avg_b = 1.0 * rs_b /node.numb_clients
            if(mode=="hard"):
                node.gmodel = (avg_w,avg_b)
            else:
                node.gmodel = ((1-self.gamma)*node.gmodel[0] + self.gamma * avg_w,
                               (1 - self.gamma) * node.gmodel[1] + self.gamma * avg_b )# (weight,bias)
            return node.gmodel
        elif(node._type.upper()=="CLIENT"): #Client
            # md = node.model.get_params() # At this time, node.model.get_params() is replaced by other client graph
            md = node.gmodel
            # print(md[0].shape,"--",md[1].shape)
            # return np.concatenate( (md[0].flatten(),md[1]), axis=0 )
            return md

    def get_hierrachical_params(self,client):
        hmd, nf = client.get_hierrachical_info1()
        # print("Normalized term:", nf)
        return (hmd[0]/nf, hmd[1]/nf) #normalized version
        # return client.get_hierrachical_info()

    def hierrachical_clustering(self):
        # if(self.Hierrchical_Method == "Weight"):
        #     weights_matrix = self.create_matrix()
        #
        # else:
        #     gradient_matrix = self.create_matrix()
        #     # gradient_matrix = np.random.rand(N_clients, Weight_dimension)
        #     model = gradient_clustering(gradient_matrix)
        model = self.run_clustering()
        self.TreeRoot = tree_construction(model, self.clients)
        print("Number of agents in tree:", self.TreeRoot.count_clients())
        print("Number of agents in level K:", self.TreeRoot.childs[0].count_clients(), self.TreeRoot.childs[1].count_clients())
        # print("Number of agents Group 1 in level K-1:", root.childs[0].childs[0].count_clients(),
        #       root.childs[0].childs[1].count_clients())

    def g_train_error_and_loss(self, gr, mode="spe" ): # mode spe: specialization, gen: generalization
        num_samples = []
        tot_correct = []
        losses = []

        self.client_model.set_params(gr.gmodel) #update parameter of group to tf.graph
        if (mode == "spe"):
            validating_clients = gr.in_clients
        else:
            validating_clients = self.clients

        for c in validating_clients:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses

    def gc_train_error_and_loss(self):
        num_samples = []
        tot_correct = []

        for c in self.clients:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)

        return np.sum(tot_correct), np.sum(tot_correct), np.sum(num_samples)


    def c_train_error_and_loss(self, i, mode="spe"): # mode spe: specialization, gen: generalization
        num_samples = []
        tot_correct = []
        losses = []
        clients_acc = []

        if (i == 0): self.client_model.set_params(self.latest_model)  # update parameter of local model initially to the shared tf.graph

        for c in self.clients:
            if (i > 0):  ## reassign to the tf.graph for testing independently to the shared tf.graph
                self.client_model.set_params(c.gmodel)
            if (mode == "spe"):
                ct, cl, ns = c.train_error_and_loss()
            else:
                ct, cl, ns = self.gc_train_error_and_loss()  # Test client as testing group approach in gen mode

            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
            clients_acc.append(ct / ns)

        # print("Training Acc Client:", clients_acc)
        # self.client_data_train[i][:] = clients_acc

        if (mode == "spe"):
            self.cs_data_train[i, :] = clients_acc
        else:
            self.cg_data_train[i, :] = clients_acc

        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses
 
    def g_test(self, gr, mode="spe" ): # mode spe: specialization, gen: generalization
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []

        self.client_model.set_params(gr.gmodel)  # update parameter of group to tf.graph
        # print("Clients in group:",self.gr.in_clients)
        if(mode =="spe"):
            validating_clients = gr.in_clients
        else:
            validating_clients = self.clients

        for c in validating_clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct


    def gc_test(self):
        num_samples = []
        tot_correct = []

        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)

        return np.sum(tot_correct), np.sum(num_samples)

    def c_test(self, i , mode="spe"): # mode spe: specialization, gen: generalization
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        clients_acc = []

        #no need to reassign client model
        if(i==0): self.client_model.set_params(self.latest_model) # update parameter of local model initially to the shared tf.graph

        for c in self.clients:
            if (i > 0): ## reassign to the tf.graph for testing independently to the shared tf.graph
                self.client_model.set_params(c.gmodel)
            if(mode=="spe"):
                ct, ns = c.test()
            else:
                ct, ns = self.gc_test()  #Test client as testing group approach in gen mode

            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            clients_acc.append(ct/ns)

        # print("Testing Acc Client:", clients_acc )
        if(mode=="spe"):
            self.cs_data_test[i,:]= clients_acc
        else:
            self.cg_data_test[i, :] = clients_acc
        # self.client_data_test.append(clients_acc)

        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct


    def evaluating_clients(self, i, mode="spe"): # mode spe: specialization, gen: generalization
        stats = self.c_test(i,mode)
        stats_train = self.c_train_error_and_loss(i,mode)
        # self.metrics.accuracies.append(stats)
        # self.metrics.train_accuracies.append(stats_train)
        test_acr = np.sum(stats[3]) * 1.0 / np.sum(stats[2])
        train_acr = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])
        tqdm.write('At round {} AvgC. testing accuracy: {}'.format(i, test_acr))
        tqdm.write('At round {} AvgC. training accuracy: {}'.format(i, train_acr ))
        # tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(
        #     stats_train[2])))
        return test_acr, train_acr

    def evaluating_groups(self,gr,i, mode="spe"): # mode spe: specialization, gen: generalization
        if(gr.parent == "Empty"):
            self.test_accs = np.zeros(K_Levels + 1)
            self.train_accs = np.zeros(K_Levels + 1)
            self.count_grs = np.zeros(K_Levels + 1)

        stats = self.g_test(gr,mode)
        stats_train = self.g_train_error_and_loss(gr,mode)
        # self.metrics.accuracies.append(stats)
        # self.metrics.train_accuracies.append(stats_train)
        test_acc =  np.sum(stats[3]) * 1.0 / np.sum(stats[2])
        train_acc = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])
        # tqdm.write('---- Test Group {} at level {} ----'.format(gr._id, gr.level))
        # tqdm.write('At round {} testing accuracy: {}'.format(i, test_acc))
        # tqdm.write('At round {} training accuracy: {}'.format(i, train_acc))
        # tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(
        #     stats_train[2])))

        self.train_accs[gr.level-1] += train_acc
        self.test_accs[gr.level-1]  += test_acc
        self.count_grs[gr.level-1]  += 1

        if (gr.childs):
            for c in gr.childs:
                if(c._type.upper()=="GROUP"):
                    self.evaluating_groups(c,i, mode)







    def show_grads(self):
        '''
        Return:
            gradients on all workers and the global gradient
        '''

        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)

        intermediate_grads = []
        samples=[]

        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            num_samples, client_grads = c.get_grads(self.latest_model)
            samples.append(num_samples)
            global_grads = np.add(global_grads, client_grads * num_samples)
            intermediate_grads.append(client_grads)

        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples))
        intermediate_grads.append(global_grads)

        return intermediate_grads

    def save(self, prox=False, lamb=0, learning_rate=0, data_set="", num_users=0, batch=0):
        alg = data_set + self.parameters['optimizer']

        if (prox == True):
            alg = alg + "_prox_" + str(lamb)
        alg = alg + "_" + str(learning_rate) + "_" + str(num_users) + "u" + "_" + str(self.batch_size) + "b"
        with h5py.File("./results/"+'{}_{}.h5'.format(alg, self.parameters['num_epochs']), 'w') as hf:
            hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
            hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
            hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
            hf.close()
        # pass

    def select_clients(self, round, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if(num_clients == len(self.clients)):
            print("All users are selected")
            return self.clients

        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round)
        return np.random.choice(self.clients, num_clients, replace=False) #, p=pk)


    def aggregate(self, wsolns, weighted=True):
        total_weight = 0.0
        base = [0]*len(wsolns[0][1])
        for (w, soln) in wsolns:  # w is the number of samples
            # Equal weights
#            if(weighted==False):
#                w=1 # Equal weights
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w*v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]
        return averaged_soln

    def aggregate_derivate(self, fsolns, weighted=True):
        total_derivative = 0.0
        base = [0]*len(fsolns[0][1])
        for (f, soln) in fsolns:  # w is the number of samples
            total_derivative += f
            for i, v in enumerate(soln):
                base[i] += f*v.astype(np.float64)

        averaged_soln = [v / total_derivative for v in base]
        return averaged_soln


