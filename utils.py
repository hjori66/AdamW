import matplotlib.pylab as plt
import torch
import random
import numpy as np


def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_Syn(adam_rt, adam_x, amsgrad_rt, amsgrad_x, epochs):

    plt.figure(figsize=(18, 6))
    plt.title('synthetic experiment on {} epochs'.format(epochs))
    plt.subplot(1, 2, 1).get_xaxis().set_visible(False)
    plt.plot(adam_rt, label='Adam')
    plt.plot(amsgrad_rt, label='AMSGrad')
    plt.grid(b=True, color='0.60', linestyle='--')
    plt.legend(fontsize=14)
    plt.ylabel('Rt/t', fontsize=14)
    plt.tick_params(axis='y', labelsize=14)

    plt.subplot(1, 2, 2).get_xaxis().set_visible(False)
    plt.plot(adam_x, label='Adam')
    plt.plot(amsgrad_x, label='AMSGrad')
    plt.grid(b=True, color='0.60', linestyle='--')
    plt.legend(fontsize=14)
    plt.ylabel('xt', fontsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.savefig('synthetic experiment.png')


def plot_CIFAR(train_loss_list, test_loss_list, opt_list, epochs):

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1).get_xaxis().set_visible(False)
    plt.title('neural network experiment on {} epochs'.format(epochs))
    for train_loss, opt in zip(train_loss_list, opt_list):
        plt.plot(train_loss, label=opt)
    plt.grid(b=True, color='0.60', linestyle='--')
    plt.legend(fontsize=14)
    plt.ylabel('Train Loss', fontsize=14)
    plt.tick_params(axis='y', labelsize=14)

    plt.subplot(1, 2, 2).get_xaxis().set_visible(False)
    for test_loss, opt in zip(test_loss_list, opt_list):
        plt.plot(test_loss, label=opt)
    plt.grid(b=True, color='0.60', linestyle='--')
    plt.legend(fontsize=14)
    plt.ylabel('Test Loss', fontsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.savefig('neural network experiment.png')
