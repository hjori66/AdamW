from model import CNN
from amsgrad import AMSGrad
from adamw import AdamW
from cifar10_dataset import dataset
from utils import fix_seed, plot_CIFAR
import torch
import argparse
from tqdm import tqdm


def main(args, model, optimizer, trainloader, testloader):
    train_loss = []
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in tqdm(range(args.epochs), desc='training...'):
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            x, y = data
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        ep_loss = running_loss/len(trainloader)
        train_loss.append(ep_loss)

    test_loss = []
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            x, y = data
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, y)
            test_loss.append(loss)

    return train_loss, test_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=128, type=float)
    parser.add_argument('--epochs', default=2000000, type=int)
    parser.add_argument('--eps', default=1e-7, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.99, type=float)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    fix_seed(args.seed)

    trainloader, testloader = dataset(args.batch_size)
    model = CNN()

    print('Processing AMSGrad...')
    amsgrad = AMSGrad(model.parameters(), lr=args.lr,
                      beta1=args.beta1, beta2=args.beta2, eps=args.eps)
    ams_train_loss, ams_test_loss = main(args, model, amsgrad, trainloader, testloader)

    print('\nProcessing SGD...')
    sgd = torch.optim.SGD(model.parameters(), lr=args.lr)
    sgd_train_loss, sgd_test_loss = main(args, model, sgd, trainloader, testloader)

    print('\nProcessing Adam...')
    adam = torch.optim.Adam(model.parameters(), lr=args.lr,
                            betas=(args.beta1, args.beta2), eps=args.eps)
    adam_train_loss, adam_test_loss = main(args, model, adam, trainloader, testloader)

    print('\nProcessing Adam with weight decay...')
    adam_wd = torch.optim.Adam(model.parameters(), lr=args.lr,
                            betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=1e-3)
    adam_wd_train_loss, adam_wd_test_loss = main(args, model, adam_wd, trainloader, testloader)

    print('\nProcessing AdaGrad...')
    adagrad = torch.optim.Adagrad(model.parameters())
    ada_train_loss, ada_test_loss = main(args, model, adagrad, trainloader, testloader)

    print('\nProcessing AdamW (with decoupled weight decay)...')
    adamw = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              betas=(args.beta1, args.beta2), eps=args.eps, decoupled_weight_decay=1e-3)
    adamw_train_loss, adamw_test_loss = main(args, model, adamw, trainloader, testloader)

    train_loss_list = [sgd_train_loss, adam_train_loss, adam_wd_train_loss, ams_train_loss, ada_train_loss, adamw_train_loss]
    test_loss_list = [sgd_test_loss, adam_test_loss, adam_wd_test_loss, ams_test_loss, ada_test_loss, adamw_test_loss]
    opt_list = ['SGD', 'Adam', 'Adam_weight_decay', 'AMSGrad', 'AdaGrad', 'AdamW']

    plot_CIFAR(train_loss_list, test_loss_list, opt_list, args.epochs)
