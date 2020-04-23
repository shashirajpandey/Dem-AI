import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
from flearn.utils.tf_utils import process_grad
from flearn.optimizer.proxsgd import PROXSGD
from .dembase import DemBase
from flearn.utils.DTree import  Node
import matplotlib.pyplot as plt

class Server(DemBase):
    def __init__(self, params, learner, dataset):

        print('Using Dem Average to Train')
        if(params["lamb"] > 0):
            self.inner_opt = PROXSGD(params['learning_rate'], params["lamb"])
        else:
            self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])

        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated Averaging'''
        print("Train using Dem Averaging")
        print('Training with {} workers ---'.format(self.clients_per_round))

        # for i in trange(self.num_rounds, desc='Round: ', ncols=120):
        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                # ============= Test each client =============
                tqdm.write('============= Test Client Models - Specialization ============= ')
                stest_acu, strain_acc = self.evaluating_clients(i,mode="spe")
                self.s_data_test.append( stest_acu )
                self.s_data_train.append(strain_acc)
                tqdm.write('============= Test Client Models - Generalization ============= ')
                gtest_acu, gtrain_acc = self.evaluating_clients(i, mode="gen")
                self.g_data_test.append(gtest_acu)
                self.g_data_train.append( gtrain_acc)

                # ============= Test root =============
                if(i>0):
                    tqdm.write('============= Test Group Models - Specialization ============= ')
                    # self.TreeRoot.print_structure()
                    self.evaluating_groups(self.TreeRoot,i,mode="spe")
                    gs_test = self.test_accs / self.count_grs
                    gs_train = self.train_accs / self.count_grs
                    self.gs_data_test.append(gs_test)
                    self.gs_data_train.append(gs_train)
                    print("AvgG. Testing performance for each level:", gs_test)
                    print("AvgG. Training performance for each level:", gs_train)
                    tqdm.write('============= Test Group Models - Generalization ============= ')
                    self.evaluating_groups(self.TreeRoot, i, mode="gen")
                    gg_test = self.test_accs/ self.count_grs
                    gg_train = self.train_accs/self.count_grs
                    self.gg_data_test.append(gg_test)
                    self.gg_data_train.append( gg_train)
                    print("AvgG. Testing performance for each level:", gg_test)
                    print("AvgG. Training performance for each level:",gg_train)



                # self.rs_glob_acc.append(np.sum(stats[3])*1.0/np.sum(stats[2]))
                # self.rs_train_acc.append(np.sum(stats_train[3])*1.0/np.sum(stats_train[2]))
                # self.rs_train_loss.append(np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2]))
                #
                # model_len = process_grad(self.latest_model).size
                # global_grads = np.zeros(model_len)
                # client_grads = np.zeros(model_len)
                # num_samples = []
                # local_grads = []
                #
                # for c in self.clients:
                #     num, client_grad = c.get_grads(model_len)
                #     local_grads.append(client_grad)
                #     num_samples.append(num)
                #     global_grads = np.add(global_grads, client_grads * num)
                # global_grads = global_grads * 1.0 / np.sum(np.asarray(num_samples))
                #
                # difference = 0
                # for idx in range(len(self.clients)):
                #     difference += np.sum(np.square(global_grads - local_grads[idx]))
                # difference = difference * 1.0 / len(self.clients)
                # tqdm.write('gradient difference: {}'.format(difference))
                #
                # # save server model
                # self.metrics.write()
                # self.save()

            # choose K clients prop to data size
            # selected_clients = self.select_clients(i, num_clients=self.clients_per_round)
            selected_clients = self.clients

            csolns = [] # buffer for receiving client solutions
            cgrads =[]
            for c in selected_clients:
            # for c in tqdm(selected_clients, desc='Client: ', leave=False, ncols=120):
                # communicate the latest model

                if(i==0):
                    c.set_params(self.latest_model)
                else:
                    #Initialize the model of client based on hierrachical GK
                    # init_w = (1 - self.beta)*(c.model.get_params()[0])
                    # init_b = (1 - self.beta)*(c.model.get_params()[1])
                    # hm = self.get_hierrachical_params(c)
                    # # print(hm)
                    # init_w += self.beta*hm[0]
                    # init_b += self.beta*hm[1]

                    # Initialize the model of client based on hierrachical GK
                    init_w = (1 - self.beta)*(c.model.get_params()[0])
                    init_b = (1 - self.beta)*(c.model.get_params()[1])
                    hm = self.get_hierrachical_params(c)
                    # print(hm)
                    init_w += self.beta*hm[0]
                    init_b += self.beta*hm[1]
                    c.set_params((init_w, init_b))
                    # c.set_params(self.TreeRoot.gmodel)
                    # c.set_params(self.latest_model)


                # solve minimization locally
                # soln, grads, stats  = c.solve_inner(
                #     self.optimizer, num_epochs=self.num_epochs, batch_size=self.batch_size)  #Local round
                #
                # # gather solutions from client
                # csolns.append(soln)
                # cgrads.append(grads)

                _, _, stats = c.solve_inner(
                    self.optimizer, num_epochs=self.num_epochs, batch_size=self.batch_size)  # Local round

                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)
            # print("First Client model:", np.sum(csolns[0][1][0]), np.sum(csolns[0][1][1]))
            if (i % 2 == 0):
                print("DEM-AI --------->>>>> Clustering")
                self.hierrachical_clustering()
                print("DEM-AI --------->>>>> Hard Update generalized model")
                self.update_generalized_model(self.TreeRoot) #hard update
                # print("Root Model:", np.sum(self.TreeRoot.gmodel[0]),np.sum(self.TreeRoot.gmodel[1]))
            else:
                # update model
                # self.latest_model = self.aggregate(csolns,weighted=True)
                print("DEM-AI --------->>>>> Soft Update generalized model")
                self.update_generalized_model(self.TreeRoot,mode="soft") #soft update
                # print("Root Model:", np.sum(self.TreeRoot.gmodel[0]),np.sum(self.TreeRoot.gmodel[1]))

        self.display_results()
    def display_results(self):
        print("Ve hinh")

        plt.clf()
        plt.figure(3)
        plt.clf()
        plt.plot( np.arange(len(self.s_data_train)), self.s_data_train, label="s_train" )
        plt.plot(np.arange(len(self.s_data_test)), self.s_data_test,label="s_test")
        plt.legend()
        plt.grid()
        plt.title("clients")

        plt.figure(4)
        plt.clf()
        plt.plot(np.arange(len(self.g_data_train)), self.g_data_train, label="g_train")
        plt.plot(np.arange(len(self.g_data_test)), self.g_data_test, label="g_test")

        plt.legend()
        plt.grid()
        plt.title("clients")

        plt.figure(5)
        plt.clf()
        plt.plot(np.arange(len(self.gs_data_train)), self.gs_data_train, label="s_train")
        plt.plot(np.arange(len(self.gs_data_test)), self.gs_data_test, label="s_test")
        # print(self.gs_data_test)

        plt.legend()
        plt.grid()
        plt.title("group")

        plt.figure(6)
        plt.clf()
        plt.plot(np.arange(len(self.gg_data_train)), self.gg_data_train, label="g_train")
        plt.plot(np.arange(len(self.gg_data_test)), self.gg_data_test, label="g_test")
        plt.legend()
        plt.grid()
        plt.title("group")

        plt.figure(7)
        plt.clf()


        # plt.plot(np.transpose(self.client_data_test))
        # for i in self.client_data_test:
        # #     plt.plot(i)
        # for cl in self.client_data_test:
        #     print(cl)
        tt = np.asarray(self.client_data_test)
        for i in range(self.N_clients):
            plt.plot(tt[:,i])
        plt.grid()
        plt.title("client test")

        print("-----------------------------||----------------")
        plt.figure(8)
        plt.clf()
        # print(self.client_data_train)
        # plt.plot(self.client_data_train)
        # for i in self.client_data_train:
        #     plt.plot(i)
        ttrain = np.asarray(self.client_data_train)
        for i in range(self.N_clients):
            plt.plot(ttrain[:, i])
        plt.grid()
        plt.title("client train")

        plt.show()
        # # final test model
        # stats = self.test()
        # # stats_train = self.train_error()
        # # stats_loss = self.train_loss()
        # stats_train = self.train_error_and_loss()
        #
        # self.metrics.accuracies.append(stats)
        # self.metrics.train_accuracies.append(stats_train)
        # tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3])*1.0/np.sum(stats[2])))
        # tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
        # # save server model
        # self.metrics.write()
        # #self.save()
        # self.save(learning_rate=self.parameters["learning_rate"])
        #
        # print("Test ACC:", self.rs_glob_acc)
        # print("Training ACC:", self.rs_train_acc)
        # print("Training Loss:", self.rs_train_loss)
