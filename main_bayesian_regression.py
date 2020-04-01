from __future__ import print_function

import os
import argparse

import torch
import numpy as np
from torch.optim import Adam
from torch.nn import functional as F

import data
import utils
import metrics
import config_bayesian as cfg
from models.BayesianModels.regression.Bayesian3Conv3FC_1D import BBB3Conv3FC_1D
from models.BayesianModels.regression.Bayesian3Liner import BBB3Liner
import GPy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getModel(net_type, inputs, outputs):
    # if (net_type == 'lenet'):
    #     return BBBLeNet(outputs,inputs)
    # elif (net_type == 'alexnet'):
    #     return BBBAlexNet(outputs, inputs)
    if (net_type == '3conv3fc'):
        return BBB3Conv3FC_1D(outputs,inputs,init_log_noise=0)
    elif(net_type == '3liner'):
        return BBB3Liner(outputs, inputs, init_log_noise=0)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')


def train_model(net, optimizer, criterion, trainloader, num_ens=1):
    net.train()
    training_loss = 0.0
    mses = []
    kl_list = []
    freq = cfg.recording_freq_per_epoch
    print(len(trainloader))
    freq = len(trainloader)//freq
    for i, (inputs, targets) in enumerate(trainloader, 1):
        cfg.curr_batch_no = i
        if i%freq==0:
            cfg.record_now = True
        else:
            cfg.record_now = False

        optimizer.zero_grad()

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = torch.zeros(inputs.shape[0], net.outputs, num_ens).to(device)

        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = net_out
        
        kl = kl / num_ens
        kl_list.append(kl.item())
        outputs = torch.mean(outputs, dim=2)
        if outputs.shape[1] == 1:
            outputs = outputs.reshape([outputs.shape[0]])

        loss = criterion(outputs, targets, net.log_noise.exp(), net.outputs, kl)
        loss.backward()
        optimizer.step()

        mses.append(metrics.mse(outputs.data, targets).cpu().data.numpy())
        training_loss += loss.cpu().data.numpy()
    return training_loss/len(trainloader), np.mean(np.array(mses)), np.mean(np.array(kl_list))


def validate_model(net, criterion, validloader, num_ens=1):
    """Calculate ensemble MSE and NLL Loss"""
    net.eval()
    valid_loss = 0.0
    mses = []

    for i, (inputs, targets) in enumerate(validloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = torch.zeros(inputs.shape[0], net.outputs, num_ens).to(device)

        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = net_out

        outputs = torch.mean(outputs, dim=2)
        if outputs.shape[1] == 1:
            outputs = outputs.reshape([outputs.shape[0]])
        loss = criterion(outputs, targets, net.log_noise.exp(), net.outputs, kl)
        mses.append(metrics.mse(outputs.data, targets).cpu().data.numpy())
        valid_loss += loss.cpu().data.numpy()

    return valid_loss/len(validloader), np.mean(np.array(mses))

def test_model(net, criterion, testloader, num_ens=1):
    """Calculate ensemble MSE and NLL Loss"""
    net.eval()
    test_loss = 0.0
    mses = []

    for i, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = torch.zeros(inputs.shape[0], net.outputs, num_ens).to(device)

        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = net_out

        outputs = torch.mean(outputs, dim=2)
        if outputs.shape[1] == 1:
            outputs = outputs.reshape([outputs.shape[0]])
        loss = criterion(outputs, targets, net.log_noise.exp(), net.outputs, kl)
        mses.append(metrics.mse(outputs.data, targets).cpu().data.numpy())
        test_loss += loss.cpu().data.numpy()

    return test_loss/len(testloader), np.mean(np.array(mses))


def test_uncertainty(net, testset, data='ccpp'):
    num_ens = 10
    # samples = []
    # for i, (inputs, targets) in enumerate(test_loader):
    #     inputs, targets = inputs.to(device), targets.to(device)
    #     outputs = torch.zeros(inputs.shape[0], net.outputs, num_ens).to(device)
    #     for j in range(num_ens):
    #         net_out, _kl = net(inputs)
    #         outputs[:, :, j] = net_out
    #     samples.append(outputs.cpu().data.numpy())
    # for k in range(len(samples)-1):
    #     samples[k+1] = np.concatenate((samples[k], samples[k+1]))
    # samples = samples[len(samples)-1]
    # means = samples.mean(axis=2)
    # means = means.reshape([means.shape[0]])
    # aleatoric = net.log_noise.exp().cpu().data.numpy()
    # epistemic = samples.var(axis=2) ** 0.5
    # epistemic = epistemic.reshape([epistemic.shape[0]])
    # total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5
    inputs = testset[:len(testset)][0]
    targets = testset[:len(testset)][1]
    # print(inputs)
    # print(targets)
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = torch.zeros(inputs.shape[0], net.outputs, num_ens).to(device)
    for j in range(num_ens):
        net_out, _kl = net(inputs)
        outputs[:, :, j] = net_out
    outputs = outputs.cpu().data.numpy()
    means = outputs.mean(axis=2)
    means = means.reshape([means.shape[0]])
    aleatoric = net.log_noise.exp().cpu().data.numpy()
    epistemic = outputs.var(axis=2) ** 0.5
    epistemic = epistemic.reshape([epistemic.shape[0]])
    total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5

    c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    plt.figure(figsize=(10, 5))
    plt.style.use('default')

    inputs = inputs.cpu().data.numpy()
    targets = targets.cpu().data.numpy()

    print("偶然误差aleatoric:")
    print(aleatoric)
    print("(样本标签, mean, Standard Deviation)")
    print(list(zip(targets, means,epistemic)))

    if data == 'uci_har':
        plt.scatter(targets,means, s=10, marker='x', color='black', alpha=0.5)
        plt.show()
        return

    for i in range(4):
        print("Dim %d: " % (i + 1))
        inputs_dim = inputs[:, i]
        targets_dim = targets
        x = np.vstack((inputs_dim, targets_dim, means, epistemic, total_unc))
        idex = np.lexsort([x[0]])
        x = x[:, idex]
        inputs_dim = x[0, :]
        targets_dim = x[1, :]
        means = x[2, :]
        epistemic = x[3, :]
        total_unc = x[4, :]

        plt.scatter(inputs_dim, targets_dim, s=10, marker='x', color='black', alpha=0.5)
        plt.plot(inputs_dim, means)
        plt.fill_between(inputs_dim, means + aleatoric, means + total_unc, color=c[0],
                         alpha=0.3, label=r'$\sigma(y^*|x^*)$')
        plt.fill_between(inputs_dim, means - total_unc, means - aleatoric, color=c[0],
                         alpha=0.3)
        plt.fill_between(inputs_dim, means - aleatoric, means + aleatoric, color=c[1],
                         alpha=0.4, label=r'$\EX[\sigma^2]^{1/2}$')
        plt.plot(inputs_dim, means, color='red', linewidth=0.1)

        # plt.xlim([inputs_dim.min(),inputs_dim.max()])
        # plt.ylim([300, 600])
        plt.xlabel('$input$', fontsize=20)
        plt.title('BBP', fontsize=20)
        plt.tick_params(labelsize=10)
        # plt.xticks(np.arange(-4, 5, 2))
        # plt.gca().set_yticklabels([])
        plt.gca().yaxis.grid(alpha=0.3)
        plt.gca().xaxis.grid(alpha=0.3)
        plt.savefig('bbp_ccpp_dim%d.pdf' %(i+1), bbox_inches='tight')
        plt.show()
        plt.cla()


def run(dataset, net_type, train=True):

    # Hyper Parameter settings
    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    test_ens = cfg.test_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size

    trainset, testset, inputs, outputs = data.getDataset_regression(dataset)

    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs).to(device)

    print(len(train_loader))
    print(len(valid_loader))
    print(len(test_loader))

    ckpt_dir = f'checkpoints/regression/{dataset}/bayesian'
    ckpt_name = f'checkpoints/regression/{dataset}/bayesian/model_{net_type}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = metrics.ELBO_regression(len(trainset)).to(device)

    if train:
        optimizer = Adam(net.parameters(), lr=lr_start)
        valid_loss_max = np.Inf
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            cfg.curr_epoch_no = epoch
            utils.adjust_learning_rate(optimizer, metrics.lr_linear(epoch, 0, n_epochs, lr_start))

            train_loss, train_mse, train_kl = train_model(net, optimizer, criterion, train_loader, num_ens=train_ens)
            valid_loss, valid_mse = validate_model(net, criterion, valid_loader, num_ens=valid_ens)

            print('Epoch: {} \tTraining Loss: {:.4f} \tTraining MSE: {:.4f} \tValidation Loss: {:.4f} \tValidation MSE: {:.4f} \ttrain_kl_div: {:.4f}'.format(
                epoch, train_loss, train_mse, valid_loss, valid_mse, train_kl))

            # save model if validation MSE has increased
            if valid_loss <= valid_loss_max:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_max, valid_loss))
                torch.save(net.state_dict(), ckpt_name)
                valid_loss_max = valid_loss

    # test saved model
    best_model = getModel(net_type, inputs, outputs).to(device)
    best_model.load_state_dict(torch.load(ckpt_name))
    test_loss, test_mse = test_model(best_model, criterion, test_loader, num_ens=test_ens)
    print('Test Loss: {:.4f} \tTest MSE: {:.4f} '.format(
            test_loss, test_mse))
    test_uncertainty(best_model, testset[:500], data='uci_har')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Bayesian Model Training")
    # 在这里指定model和数据
    parser.add_argument('--net_type', default='3conv3fc', type=str, help='model')
    parser.add_argument('--dataset', default='uci_har', type=str, help='dataset = [ccpp, uci_har]')
    args = parser.parse_args()

    if cfg.record_mean_var:
        mean_var_dir = f"checkpoints/regression/{args.dataset}/bayesian/{args.net_type}/"
        cfg.mean_var_dir = mean_var_dir
        if not os.path.exists(mean_var_dir):
            os.makedirs(mean_var_dir, exist_ok=True)
        for file in os.listdir(mean_var_dir):
            os.remove(mean_var_dir + file)

    run(args.dataset, args.net_type, train=False)
