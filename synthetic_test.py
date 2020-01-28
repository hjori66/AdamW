import torch
import numpy as np
import matplotlib.pylab as plt
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
from amsgrad import AMSGrad
from utils import fix_seed, plot_Syn
import argparse


def train_synthetic(x, optimizer, iter):
    avg_rt = []
    x_list = []

    xmax = Variable(torch.Tensor([1.0]), requires_grad=True)
    xmin = Variable(torch.Tensor([-1.0]), requires_grad=True)

    rt = 0
    for step in tqdm(range(1, iter+1)):
        if x > 1.0:
            x = xmax
        if x < -1.0:
            x = xmin

        if step % 101 == 1:
            loss = x*1010
            min = -1010
        else:
            loss = -10*x
            min = 10

        rt += (loss.item() - min)
        avg_Rt = rt/step

        if step % (iter/100) == 0:
            avg_rt.append(avg_Rt)
            x_list.append(x.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return avg_rt, x_list


def main(args):
    x = Variable(torch.Tensor([0.5]), requires_grad=True)

    print('\nProcessing Adam...')
    adam = torch.optim.Adam([x], lr=args.lr, betas=(0.9, 0.99), eps=1e-8)
    adam_rt, adam_x = train_synthetic(x, adam, args.epochs)

    print('Processing AMSGrad...')
    amsgrad = AMSGrad([x], lr=args.lr, beta1=0.9, beta2=0.99)
    amsgrad_rt, amsgrad_x = train_synthetic(x, amsgrad, args.epochs)

    plot_Syn(adam_rt, adam_x, amsgrad_rt, amsgrad_x, args.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epochs', default=6000000, type=int)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    assert args.lr > 0, 'learning rate must be greater than 0'
    assert args.epochs > 0, 'epochs must be greater than 0'
    fix_seed(args.seed)
    main(args)
