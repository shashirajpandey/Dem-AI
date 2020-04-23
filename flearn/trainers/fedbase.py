import numpy as np
import tensorflow as tf
from tqdm import tqdm

from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad
import h5py
import matplotlib.pyplot as plt

class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        # transfer parameters to self

        for key, val in params.items(): setattr(self, key, val);
        self.parameters = params
        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(*params['model_params'], self.inner_opt, self.seed)
        #initilzation of clients
        self.clients = self.setup_clients(dataset, self.client_model)
        self.N_clients = len(self.clients)
        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = self.client_model.get_params()

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc = [], [], []
        self.global_data_test = []  # generalization of global test accuracy
        self.global_data_train = []  # specialization of global train accuracy
        self.cs_data_test = np.zeros((self.num_rounds, self.N_clients))
        self.cs_data_train = np.zeros((self.num_rounds, self.N_clients))
        self.cg_data_test = np.zeros((self.num_rounds, self.N_clients))
        self.cg_data_train = np.zeros((self.num_rounds, self.N_clients))
        self.cg_avg_data_test = []  # avg generalization client accuracy test
        self.cg_avg_data_train = []  # avg generalization client accuracy train
        self.cs_avg_data_test = []  # avg specialization client test accuracy
        self.cs_avg_data_train = []  # avg specialization client train accuracy
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
        all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
        return all_clients

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.clients:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses


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
 
  
    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            # print("Acc Client", c.id, ":", ct / ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

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
#             w=1
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
        if (mode == "spe"):
            self.cs_data_test[i, :] = clients_acc
        else:
            self.cg_data_test[i, :] = clients_acc
        # print("Testing Acc Client:", clients_acc )
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

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
        if (mode == "spe"):
            self.cs_data_train[i, :] = clients_acc
        else:
            self.cg_data_train[i, :] = clients_acc
        # print("Training Acc Client:", clients_acc)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses


    def evaluating_clients(self, i, mode="spe"): # mode spe: specialization, gen: generalization
        stats = self.c_test(i,mode)
        stats_train = self.c_train_error_and_loss(i,mode)
        # self.metrics.accuracies.append(stats)
        # self.metrics.train_accuracies.append(stats_train)

        test_acr = np.sum(stats[3]) * 1.0 / np.sum(stats[2])
        train_acr = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])
        tqdm.write('At round {} AvgC. testing accuracy: {}'.format(i, test_acr))
        tqdm.write('At round {} AvgC. training accuracy: {}'.format(i, train_acr))
        # tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(
        #     stats_train[2])))
        return test_acr, train_acr

    def evaluating_global(self,i):
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        gl_test = np.sum(stats[3])*1.0/np.sum(stats[2])
        gl_train = np.sum(stats_train[3])*1.0/np.sum(stats_train[2])
        self.global_data_test.append(gl_test)
        self.global_data_train.append(gl_train)
        tqdm.write('At round {} global testing accuracy: {}'.format(i, gl_test))
        tqdm.write('At round {} global training accuracy: {}'.format(i, gl_train))
        tqdm.write('At round {} global training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])))

