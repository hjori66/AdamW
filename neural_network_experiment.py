from model import CNN
from amsgrad import AMSGrad
from adamw import AdamW
from cifar_dataset import cifar10_dataset
from utils import fix_seed, plot_CIFAR
import torch
import argparse
import pickle
from tqdm import tqdm


def main(args, model, optimizer, trainloader, testloader):
    train_loss = []
    test_loss = []
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in tqdm(range(args.epochs), desc='training...'):
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            x, y = data
            x = x.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        ep_loss = running_loss/len(trainloader)
        train_loss.append(ep_loss)

        with torch.no_grad():
            total_test_loss = 0
            for idx, data in enumerate(testloader):
                x, y = data
                x = x.cuda()
                y = y.cuda()
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, y)
                total_test_loss += loss.item()
            test_loss.append(total_test_loss / (idx+1))

    with open('results/loss/' + args.optimizer + '_train_loss.txt', 'wb') as f:
        pickle.dump(train_loss, f)

    with open('results/loss/' + args.optimizer + '_test_loss.txt', 'wb') as f:
        pickle.dump(test_loss, f)

    return train_loss, test_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=128, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--eps', default=1e-7, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.99, type=float)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--optimizer', default='draw_plot', type=str)
    args = parser.parse_args()
    fix_seed(args.seed)

    trainloader, testloader = cifar10_dataset(args.batch_size)

    model = torch.hub.load('pytorch/vision:v0.5.0', 'vgg11', pretrained=True)
    model.cuda()

    if args.optimizer == 'SGD':
        print('\nProcessing SGD...')
        sgd = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=args.momentum)
        sgd_train_loss, sgd_test_loss = main(args, model, sgd, trainloader, testloader)

    elif args.optimizer == 'Adam':
        print('\nProcessing Adam...')
        adam = torch.optim.Adam(model.parameters(), lr=args.lr,
                                betas=(args.beta1, args.beta2), eps=args.eps)
        adam_train_loss, adam_test_loss = main(args, model, adam, trainloader, testloader)

    elif args.optimizer == 'Adam_weight_decay':
        print('\nProcessing Adam with weight decay...')
        adam_wd = torch.optim.Adam(model.parameters(), lr=args.lr,
                                betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
        adam_wd_train_loss, adam_wd_test_loss = main(args, model, adam_wd, trainloader, testloader)

    elif args.optimizer == 'AdamW':
        print('\nProcessing AdamW (with decoupled weight decay)...')
        adamw = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=1.0)
        adamw_train_loss, adamw_test_loss = main(args, model, adamw, trainloader, testloader)

    elif args.optimizer == 'AMSGrad':
        print('Processing AMSGrad...')
        amsgrad = AMSGrad(model.parameters(), lr=args.lr,
                          beta1=args.beta1, beta2=args.beta2, eps=args.eps)
        ams_train_loss, ams_test_loss = main(args, model, amsgrad, trainloader, testloader)

    elif args.optimizer == 'AdaGrad':
        print('\nProcessing AdaGrad...')
        adagrad = torch.optim.Adagrad(model.parameters())
        ada_train_loss, ada_test_loss = main(args, model, adagrad, trainloader, testloader)

    else:
        # opt_list = ['SGD', 'Adam', 'Adam_weight_decay', 'AdamW', 'AMSGrad', 'AdaGrad']
        opt_list = ['Adam']
        train_loss_list = []
        test_loss_list = []
        for opt in opt_list:
            with open('results/loss/' + opt + '_train_loss.txt', 'rb') as f:
                train_loss = pickle.load(f)
            train_loss_list.append(train_loss)

            with open('results/loss/' + opt + '_test_loss.txt', 'rb') as f:
                test_loss = pickle.load(f)
            test_loss_list.append(test_loss)

        plot_CIFAR(train_loss_list, test_loss_list, opt_list, args.epochs)
