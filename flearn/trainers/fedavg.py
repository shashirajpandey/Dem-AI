import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
from flearn.utils.tf_utils import process_grad
from flearn.optimizer.proxsgd import PROXSGD
from .fedbase import BaseFedarated
import matplotlib.pyplot as plt

class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated Average to Train')
        if(params["lamb"] > 0):
            self.inner_opt = PROXSGD(params['learning_rate'], params["lamb"])
        else:
            self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated Averaging'''
        print("Train using Federated Averaging")
        print('Training with {} workers ---'.format(self.clients_per_round))
        # for i in trange(self.num_rounds, desc='Round: ', ncols=120):
        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                # ============= Test each client =============
                tqdm.write('============= Test Client Models - Specialization ============= ')
                stest_acu, strain_acc = self.evaluating_clients(i, mode="spe")
                self.s_data_test.append(stest_acu)
                self.s_data_train.append(strain_acc)
                tqdm.write('============= Test Client Models - Generalization ============= ')
                gtest_acu, gtrain_acc = self.evaluating_clients(i, mode="gen")
                self.g_data_test.append(gtest_acu)
                self.g_data_train.append(gtrain_acc)
                # tqdm.write('============= Test Client Models - Specialization ============= ')
                # self.evaluating_clients(i,mode="spe")
                # tqdm.write('============= Test Client Models - Generalization ============= ')
                # self.evaluating_clients(i, mode="gen")
                tqdm.write('============= Test Global Models  ============= ')
                self.evaluating_global(i)


        #     # test model
        #     if i % self.eval_every == 0:
        #         stats = self.test()
        #         stats_train = self.train_error_and_loss()
        #         self.metrics.accuracies.append(stats)
        #         self.metrics.train_accuracies.append(stats_train)
        #         tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3])*1.0/np.sum(stats[2])))
        #         tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
        #         tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])))

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
            for c in selected_clients:
            # for c in tqdm(selected_clients, desc='Client: ', leave=False, ncols=120):
                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally
                soln, grads, stats  = c.solve_inner(
                    self.optimizer, num_epochs=self.num_epochs, batch_size=self.batch_size)
                c.gmodel=soln[1]
                # gather solutions from client
                csolns.append(soln)

                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)
            # print("First Client model:", csolns[0][1])
            # print("First Client model:", np.sum(csolns[0][1][0]))
            # update model
            self.latest_model = self.aggregate(csolns,weighted=True)
            # print("Averaging model:", self.latest_model )
            # print("Averaging model:", np.sum(self.latest_model[0]))

        # final test model
        stats = self.test()
        # stats_train = self.train_error()
        # stats_loss = self.train_loss()
        stats_train = self.train_error_and_loss()

        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3])*1.0/np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
        # save server model
        self.metrics.write()
        #self.save()
        self.save(learning_rate=self.parameters["learning_rate"])

        print("Test ACC:", self.rs_glob_acc)
        print("Training ACC:", self.rs_train_acc)
        print("Training Loss:", self.rs_train_loss)
        self.display_results()
    def display_results(self):
        print("FED--------------> Plotting")

        avg_root_test = np.asarray(self.root_data_test)
        avg_root_train = np.asarray(self.root_data_train)
        plt.clf()
        plt.figure(3)
        plt.clf()
        plt.plot(avg_root_train, label="root train", linestyle="--")
        plt.plot(avg_root_test, label="root test", linestyle="--")
        plt.plot(np.arange(len(self.s_data_train)), self.s_data_train, label="s_train")
        plt.plot(np.arange(len(self.s_data_test)), self.s_data_test, label="s_test")
        plt.legend()
        plt.grid()
        plt.title("AVG Clients Specialization Accuracy")

        plt.figure(4)
        plt.clf()
        plt.plot(avg_root_train, label="root train", linestyle="--")
        plt.plot(avg_root_test, label="root test", linestyle="--")
        plt.plot(np.arange(len(self.g_data_train)), self.g_data_train, label="g_train")
        plt.plot(np.arange(len(self.g_data_test)), self.g_data_test, label="g_test")
        plt.legend()
        plt.grid()
        plt.title("AVG Clients Generalization Accuracy")

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
        #
        plt.figure(7)
        plt.clf()
        # print(self.cs_data_test)
        # print("-----------------------------||----------------")
        # plt.plot(np.transpose(self.client_data_test))
        # for i in self.client_data_test:
        #     plt.plot(i)
        plt.plot(avg_root_test, linestyle="--", label="root test")
        plt.plot(self.cs_data_test)
        plt.grid()
        plt.title("Testing Client Specialization ")

        plt.figure(8)
        plt.clf()
        # print(self.cs_data_train)
        plt.plot(avg_root_train, linestyle="--", label="root train")
        plt.plot(self.cs_data_train)
        # for i in self.client_data_train:
        #     plt.plot(i)
        plt.grid()
        plt.title("Training Client Specialization ")

        plt.figure(9)
        plt.clf()
        # print(self.cs_data_test)
        # print("-----------------------------||----------------")
        # plt.plot(np.transpose(self.client_data_test))
        # for i in self.client_data_test:
        #     plt.plot(i)
        plt.plot(self.cg_data_test)
        # plt.plot(avg_root_train, label="root train")
        plt.plot(avg_root_test, linestyle="--", label="root test")
        plt.grid()
        plt.title("Testing Client Generalization ")

        plt.figure(10)
        plt.clf()
        # print(self.cs_data_train)
        plt.plot(self.cg_data_train)
        # plt.plot(avg_root_train, label="root train")
        plt.plot(avg_root_train, linestyle="--", label="root train")
        # plt.plot(avg_root_test,linestyle="--", label="root test")
        # for i in self.client_data_train:
        #     plt.plot(i)
        plt.grid()
        plt.title("Training Client Generalization ")

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
