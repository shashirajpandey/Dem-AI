import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
import tensorflow as tf
from flearn.utils.plot_utils import plot_summary_two_figures, plot_summary_one_figure2, plot_summary_three_figures, plot_summary_three_figures_batch, plot_summary_mnist, plot_summary_nist
from flearn.utils.model_utils import read_data
from clustering.Setting import *
import data.fmnist.data

# # GLOBAL PARAMETERS
OPTIMIZERS = ['fedavg', 'fedprox', 'fedsgd', 'fedfedl']
#
DATASETS1 = ['nist', 'mnist', 'fmnist']  # NIST is EMNIST in the paper

MODEL_PARAMS = {
    'sent140.bag_dnn': (2,),  # num_classes
    'sent140.stacked_lstm': (25, 2, 100),  # seq_len, num_classes, num_hidden
    # seq_len, num_classes, num_hidden
    'sent140.stacked_lstm_no_embeddings': (25, 2, 100),
    # num_classes, should be changed to 62 when using EMNIST
    'nist.mclr': (62,),
    'nist.cnn': (62,),
    'mnist.mclr': (10,),  # num_classes
    'mnist.cnn': (10,),  # num_classes
    'cifar100.mclr': (100,),  # num_classes
    'cifar100.cnn': (100,),  # num_classes
    'fmnist.mclr': (10,),
    'fmnist.cnn': (10,),
    'shakespeare.stacked_lstm': (80, 80, 256),  # seq_len, emb_dim, num_hidden
    'synthetic.mclr': (10, )  # num_classes
}


def read_options(num_users=5, loc_ep=10, Numb_Glob_Iters=100, lamb=0, learning_rate=0.01,hyper_learning_rate= 0.01,
                 alg='fedprox', weight=True, batch_size=0, dataset="mnist", model="cnn.py"):
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',
                        help='name of optimizer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default=alg)  # fedavg, fedprox
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS1,
                        default=dataset)
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default=model)  # 'stacked_lstm.py'
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=Numb_Glob_Iters)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=num_users)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=batch_size
                        )  # 0 is full dataset
    parser.add_argument('--num_epochs',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=loc_ep)
    parser.add_argument('--learning_rate',
                        help='learning rate for inner solver;',
                        type=float,
                        default=learning_rate)  # 0.003
    parser.add_argument('--hyper_learning_rate',
                        help='learning rate for inner solver;',
                        type=float,
                        default=hyper_learning_rate)  # 0.001
    parser.add_argument('--mu',
                        help='constant for prox;',
                        type=float,
                        default=0.)  # 0.01
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--weight',
                        help='enable weight value;',
                        type=int,
                        default=weight)
    parser.add_argument('--lamb',
                        help='Penalty value for proximal term;',
                        type=int,
                        default=lamb)

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))

    # Set seeds
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    tf.set_random_seed(123 + parsed['seed'])

    # load selected model
    # all synthetic datasets use the same model
    if parsed['dataset'].startswith("synthetic"):
        model_path = '%s.%s.%s.%s' % (
            'flearn', 'models', 'synthetic', parsed['model'])
    else:
        model_path = '%s.%s.%s.%s' % (
            'flearn', 'models', parsed['dataset'], parsed['model'])

    # mod = importlib.import_module(model_path)
    if MODEL_TYPE == "cnn": #"cnn" or "mclr"
        if(DATASET=="mnist"):
            import flearn.models.mnist.cnn as cnn
        elif(DATASET=="cifar100"):
            import flearn.models.cifar100.cnn as cnn
        elif(DATASET == "fmnist"):
            import flearn.models.fmnist.cnn as cnn
        mod = cnn
    else:
        if (DATASET == "mnist"):
            import flearn.models.mnist.mclr as mclr
        elif (DATASET == "cifar100"):
            import flearn.models.cifar100.mclr as mclr
        elif (DATASET == "fmnist"):
            import flearn.models.fmnist.mclr as mclr
        mod = mclr

    learner_model = getattr(mod, 'Model')

    # load selected trainer
    alg = parsed['optimizer']
    if (alg== "demprox"):
        alg = "demavg"
    elif (alg== "demlearn-p"):
        alg = "demlearn"
    elif(alg=="fedprox"):
        alg = "fedavg"
    opt_path = 'flearn.trainers.%s' % alg


    mod = importlib.import_module(opt_path)
    trainer = getattr(mod, 'Server')

    # add selected model parameter
    parsed['model_params'] = MODEL_PARAMS['.'.join(
        model_path.split('.')[2:-1])]
    # parsed['model_params'] = MODEL_PARAMS['mnist.mclr']

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()):
        print(fmtString % keyPair)

    return parsed, learner_model, trainer


def main(num_users=5, loc_ep=10, Numb_Glob_Iters=100, lamb=0, learning_rate=0.01,hyper_learning_rate= 0.01, alg='fedprox', weight=True, batch_size=0, dataset="mnist"):
    # suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)
    model = MODEL_TYPE+".py"
    # if(dataset == "cifar100"):
    #     learning_rate = 0.001
    # # parse command line arguments
    options, learner_model, trainer = read_options(
        num_users, loc_ep, Numb_Glob_Iters, lamb, learning_rate,hyper_learning_rate, alg, weight, batch_size, dataset, model)

    # read data
    train_path = os.path.join('data', options['dataset'], 'data', 'train')
    test_path = os.path.join('data', options['dataset'], 'data', 'test')
    dataset = read_data(train_path, test_path)

    # call appropriate trainer
    t = trainer(options, learner_model, dataset)
    t.train()


if __name__ == '__main__':
    lamb_value=0
    learning_rate = 0.01
    hyper_learning_rate = 0.2
    local_ep = 10               # Number of local iterations
    batch_size = 10


    number_users = N_clients #100
    number_global_iters = NUM_GLOBAL_ITERS
    number_users = int(N_clients)

    if(READ_DATASET ==False):
        print("Generate ", str.upper(DATASET), "  Dataset with ", number_users, " users")
        if(number_users==50):
            exec(open( "./data/" + DATASET + "/generate_niid_50users.py").read())
        elif(number_users==100):
            exec(open( "./data/" + DATASET +"/generate_niid_100users.py").read())

    main(num_users=number_users, loc_ep=local_ep, Numb_Glob_Iters=number_global_iters, lamb=lamb_value,
         learning_rate=learning_rate, hyper_learning_rate=hyper_learning_rate, alg=RUNNING_ALG,
         batch_size=batch_size, dataset=DATASET)

    print("-- FINISH -- :",)
