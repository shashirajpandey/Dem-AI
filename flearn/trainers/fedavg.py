from tqdm import tqdm
import tensorflow as tf
from flearn.optimizer.pgd import PerturbedGradientDescent
from .fedbase import BaseFedarated
from utils.data_plot_mnist import *

class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated Average to Train')
        if(params['optimizer'] == "fedprox"):
            self.alg = "FEDPROX"
            print('Using FedProx to Train')
            mu = 0.005  #0.005: faster but less smooth vs 0.01: smoother but slower
            # self.inner_opt = PROXSGD(params['learning_rate'], params["lamb"])
            self.inner_opt = PerturbedGradientDescent(params['learning_rate'], mu)
        elif (params['optimizer'] == "fedavg"):
            self.alg = "FEDAVG"
            print('Using FedAvg to Train')
            self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated Averaging or Federated Proximal'''
        print("Train using " + self.alg)
        print('Training with {} workers ---'.format(self.clients_per_round))
        # for i in trange(self.num_rounds, desc='Round: ', ncols=120):
        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                # ============= Test each client =============
                tqdm.write('============= Test Client Models - Specialization ============= ')
                stest_acu, strain_acc = self.evaluating_clients(i, mode="spe")
                self.cs_avg_data_test.append(stest_acu)
                self.cs_avg_data_train.append(strain_acc)
                tqdm.write('============= Test Client Models - Generalization ============= ')
                gtest_acu, gtrain_acc = self.evaluating_clients(i, mode="gen")
                self.cg_avg_data_test.append(gtest_acu)
                self.cg_avg_data_train.append(gtrain_acc)
                tqdm.write('============= Test Global Models  ============= ')
                self.evaluating_global(i)

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
        # self.save(learning_rate=self.parameters["learning_rate"])

        print("Test ACC:", self.rs_glob_acc)
        print("Training ACC:", self.rs_train_acc)
        print("Training Loss:", self.rs_train_loss)
        self.save_results()

    def save_results(self):

        write_file(file_name=rs_file_path, root_test=self.global_data_test, root_train=self.global_data_train,
                   cs_avg_data_test=self.cs_avg_data_test, cs_avg_data_train=self.cs_avg_data_train,
                   cg_avg_data_test=self.cg_avg_data_test, cg_avg_data_train=self.cg_avg_data_train,
                   cs_data_test=self.cs_data_test, cs_data_train=self.cs_data_train, cg_data_test=self.cg_data_test,
                   cg_data_train=self.cg_data_train, N_clients=[N_clients])
        plot_from_file()

    # def display_results(self):
    #     # print("FED-AVG --------------> Plotting")
    #     alg_name=self.alg+"_"
    #
    #     global_train = np.asarray(self.global_data_train)
    #     global_test = np.asarray(self.global_data_test)
    #
    #     plt.figure(3)
    #     plt.clf()
    #     plt.plot(global_train, label="Global_train", linestyle="--")
    #     plt.plot(global_test, label="Global_test", linestyle="--")
    #     plt.plot(np.arange(len(self.cs_avg_data_train)),  self.cs_avg_data_train, linestyle="-", label="Client_spec_train")
    #     plt.plot(np.arange(len(self.cs_avg_data_test)), self.cs_avg_data_test, linestyle="-", label="Client_spec_test")
    #     plt.plot(np.arange(len(self.cg_avg_data_train)), self.cg_avg_data_train, linestyle="-", label="Client_gen_train")
    #     plt.plot(np.arange(len(self.cg_avg_data_test)), self.cg_avg_data_test, linestyle="-", label="Client_gen_test")
    #     plt.legend()
    #     plt.xlabel("Global Rounds")
    #     plt.ylim(0, 1.02)
    #     plt.grid()
    #     plt.title("AVG Clients Model (Spec-Gen) Accuracy")
    #     plt.savefig(PLOT_PATH + alg_name + "AVGC_Spec_Gen.pdf")
    #
    #     # plt.figure(3)
    #     # plt.clf()
    #     # plt.plot(global_train, label="Global_train", linestyle="--")
    #     # plt.plot(np.arange(len(self.cs_avg_data_train)), self.cs_avg_data_train, label="Client_spec_train")
    #     # plt.plot(np.arange(len(self.cg_avg_data_train)), self.cg_avg_data_train, label="Client_gen_train")
    #     # plt.legend()
    #     # plt.xlabel("Global Rounds")
    #     # plt.ylim(0, 1.02)
    #     # plt.grid()
    #     # plt.title("AVG Clients Model (Spec-Gen) Training Accuracy")
    #     # plt.savefig(PLOT_PATH + alg_name+"AVGC_Spec_Gen_Training.pdf")
    #     #
    #     # plt.figure(4)
    #     # plt.clf()
    #     # plt.plot(global_test, label="Global_test", linestyle="--")
    #     # plt.plot(np.arange(len(self.cs_avg_data_test)), self.cs_avg_data_test, label="Client_spec_test")
    #     # plt.plot(np.arange(len(self.cg_avg_data_test)), self.cg_avg_data_test, label="Client_gen_test")
    #     # plt.legend()
    #     # plt.xlabel("Global Rounds")
    #     # plt.ylim(0, 1.02)
    #     # plt.grid()
    #     # plt.title("AVG Clients Model (Spec-Gen) Testing Accuracy")
    #     # plt.savefig(PLOT_PATH + alg_name+"AVGC_Spec_Gen_Testing.pdf")
    #
    #     plt.figure(7)
    #     plt.clf()
    #     plt.plot(global_test, linestyle="--", label="root test")
    #     plt.plot(self.cs_data_test)
    #     plt.xlabel("Global Rounds")
    #     plt.ylim(0, 1.02)
    #     plt.grid()
    #     plt.title("Testing Client Specialization")
    #     plt.savefig(PLOT_PATH + alg_name+"C_Spec_Testing.pdf")
    #
    #     plt.figure(8)
    #     plt.clf()
    #     plt.plot(global_train, linestyle="--", label="root train")
    #     plt.plot(self.cs_data_train)
    #     plt.legend()
    #     plt.xlabel("Global Rounds")
    #     plt.ylim(0, 1.02)
    #     plt.grid()
    #     plt.title("Training Client Specialization")
    #     plt.savefig(PLOT_PATH + alg_name+"C_Spec_Training.pdf")
    #
    #     plt.figure(9)
    #     plt.clf()
    #     plt.plot(self.cg_data_test)
    #     plt.plot(global_test, linestyle="--", label="root test")
    #     plt.legend()
    #     plt.xlabel("Global Rounds")
    #     plt.ylim(0, 1.02)
    #     plt.grid()
    #     plt.title("Testing Client Generalization")
    #     plt.savefig(PLOT_PATH + alg_name+"C_Gen_Testing.pdf")
    #
    #     plt.figure(10)
    #     plt.clf()
    #     plt.plot(self.cg_data_train)
    #     plt.plot(global_train, linestyle="--", label="root train")
    #     plt.legend()
    #     plt.xlabel("Global Rounds")
    #     plt.ylim(0, 1.02)
    #     plt.grid()
    #     plt.title("Training Client Generalization ")
    #     plt.savefig(PLOT_PATH + alg_name+"C_Gen_Training.pdf")
    #
    #     plt.show()
    #
    #     print("** Summary Results: ---- Training ----")
    #     print("AVG Clients Specialization - Training:",self.cs_avg_data_train)
    #     print("AVG Clients Generalization - Training::",self.cg_avg_data_train)
    #     print("Global performance - Training:",global_train)
    #     print("** Summary Results: ---- Testing ----")
    #     print("AVG Clients Specialization - Testing:", self.cs_avg_data_test)
    #     print("AVG Clients Generalization - Testing:", self.cg_avg_data_test)
    #     print("Global performance - Testing:", global_test)

