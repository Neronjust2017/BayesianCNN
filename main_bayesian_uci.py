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
from models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.BayesianLeNet import BBBLeNet
from models.BayesianModels.Bayesian3Conv3FC_1D import BBB3Conv3FC_1D
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import colors

from tensorboardX import SummaryWriter

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getModel(net_type, inputs, outputs):
    if (net_type == 'lenet'):
        return BBBLeNet(outputs,inputs)
    elif (net_type == 'alexnet'):
        return BBBAlexNet(outputs, inputs)
    elif (net_type == '3conv3fc'):
        return BBB3Conv3FC(outputs,inputs)
    elif (net_type == '3conv3fc_1d'):
        return BBB3Conv3FC_1D(outputs,inputs)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')


def train_model(net, optimizer, criterion, trainloader, num_ens=1):
    net.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    freq = cfg.recording_freq_per_epoch
    print(len(trainloader))
    freq = len(trainloader)//freq
    for i, (inputs, labels) in enumerate(trainloader, 1):     # 第 i 个 batch
        cfg.curr_batch_no = i
        if i%freq==0:
            cfg.record_now = True
        else:
            cfg.record_now = False

        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)

        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)
        
        kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = utils.logmeanexp(outputs, dim=2)

        loss = criterion(log_outputs, labels, kl)
        loss.backward()
        optimizer.step()

        accs.append(metrics.acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()
    return training_loss/len(trainloader), np.mean(accs), np.mean(kl_list)


def validate_model(net, criterion, validloader, num_ens=1):
    """Calculate ensemble accuracy and NLL Loss"""
    net.eval()
    valid_loss = 0.0
    accs = []

    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

        log_outputs = utils.logmeanexp(outputs, dim=2)
        valid_loss += criterion(log_outputs, labels, kl).item()
        accs.append(metrics.acc(log_outputs, labels))

    return valid_loss/len(validloader), np.mean(accs)

def test_model(net, criterion, testloader, num_ens=10):
    """Calculate ensemble accuracy and NLL Loss"""
    net.eval()
    test_loss = 0.0
    accs = []

    for i, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

        log_outputs = utils.logmeanexp(outputs, dim=2)
        test_loss += criterion(log_outputs, labels, kl).item()
        accs.append(metrics.acc(log_outputs, labels))

    return test_loss / len(testloader), np.mean(accs)

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

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs).to(device)

    ckpt_dir = f'checkpoints/{dataset}/bayesian'
    ckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}.pt'


    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = metrics.ELBO(len(trainset)).to(device)

    if train:
        optimizer = Adam(net.parameters(), lr=lr_start)
        valid_loss_max = np.Inf
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            cfg.curr_epoch_no = epoch
            utils.adjust_learning_rate(optimizer, metrics.lr_linear(epoch, 0, n_epochs, lr_start))

            train_loss, train_acc, train_kl = train_model(net, optimizer, criterion, train_loader, num_ens=train_ens)
            valid_loss, valid_acc = validate_model(net, criterion, valid_loader, num_ens=valid_ens)

            print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
                epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl))
            print(
                'Training Loss: {:.4f} \tTraining Likelihood Loss: {:.4f} \tTraining Kl Loss: {:.4f}'.format(
                    train_loss, train_loss-train_kl,train_kl))

            # save model if validation accuracy has increased
            if valid_loss <= valid_loss_max:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_max, valid_loss))
                torch.save(net.state_dict(), ckpt_name)
                valid_loss_max = valid_loss

    # test saved model
    best_model = getModel(net_type, inputs, outputs).to(device)
    best_model.load_state_dict(torch.load(ckpt_name))
    test_loss, test_acc = test_model(best_model, criterion, test_loader, num_ens=test_ens)
    print('Test Loss: {:.4f} \tTest Accuracy: {:.4f} '.format(
            test_loss, test_acc))
    print('Test uncertainities:')
    test_uncertainities(best_model, test_loader, num_ens=10)

def give_uncertainities(net, datas, num_ens=10):
    net.eval()
    inputs = datas.to(device)
    outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
    for j in range(num_ens):
        net_out, _kl = net(inputs)
        outputs[:,:,j] = F.log_softmax(net_out, dim=1).data
    return np.asarray(outputs.cpu())
    #mean = torch.mean(torch.stack(yhats), 0)
    #return np.argmax(mean, axis=1)

def test_batch_uncertainities(net, datas, labels, num_ens, plot=True):
    y = give_uncertainities(net, datas, num_ens)
    predicted_for_images = 0
    correct_predictions = 0

    for i in range(len(labels)):

        if (plot):
            print("Real: ", labels[i].item())
            fig, axs = plt.subplots(1, 6, sharey=True, figsize=(20, 2))

        all_digits_prob = []

        highted_something = False

        for j in range(net.num_classes):

            highlight = False

            histo = []
            histo_exp = []

            for z in range(y.shape[2]):
                histo.append(y[i][j][z])
                histo_exp.append(np.exp(y[i][j][z]))

            prob = np.percentile(histo_exp, num_ens//2)  # sampling median probability

            if (prob > 0.2):  # select if network thinks this sample is 20% chance of this being a label
                highlight = True  # possibly an answer

            all_digits_prob.append(prob)

            if (plot):
                N, bins, patches = axs[j].hist(histo, bins=8, color="lightgray", lw=0,
                                               weights=np.ones(len(histo)) / len(histo), density=False)
                axs[j].set_title(str(j) + " (" + str(round(prob, 2)) + ")")

            if (highlight):

                highted_something = True

                if (plot):

                    # We'll color code by height, but you could use any scalar
                    fracs = N / N.max()

                    # we need to normalize the data to 0..1 for the full range of the colormap
                    norm = colors.Normalize(fracs.min(), fracs.max())

                    # Now, we'll loop through our objects and set the color of each accordingly
                    for thisfrac, thispatch in zip(fracs, patches):
                        color = plt.cm.viridis(norm(thisfrac))
                        thispatch.set_facecolor(color)

        if (plot):
            plt.show()

        predicted = np.argmax(all_digits_prob)

        if (highted_something):
            predicted_for_images += 1
            if (labels[i].item() == predicted):
                if (plot):
                    print("Correct")
                correct_predictions += 1.0
            else:
                if (plot):
                    print("Incorrect :()")
        else:
            if (plot):
                print("Undecided.")

        # if (plot):
        #     imshow(images[i].squeeze())

    if (plot):
        print("Summary")
        print("Total images: ", len(labels))
        print("Predicted for: ", predicted_for_images)
        print("Accuracy when predicted: ", correct_predictions / predicted_for_images)

    return len(labels), correct_predictions, predicted_for_images

# Prediction when network can decide not to predict
def test_uncertainities(net, test_loader, num_ens=10, batchs=1):
    print('Prediction when network can refuse')
    correct = 0
    total = 0
    total_predicted_for = 0
    for j, data in enumerate(test_loader):
        if j == batchs:
            return
        datas, labels = data
        total_minibatch, correct_minibatch, predictions_minibatch = test_batch_uncertainities(net, datas, labels, num_ens, plot=True)
        total += total_minibatch
        correct += correct_minibatch
        total_predicted_for += predictions_minibatch

    print("Total datas: ", total)
    print("Skipped: ", total - total_predicted_for)
    print("Accuracy when made predictions: %d %%" % (100 * correct / total_predicted_for))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Bayesian Model Training")
    parser.add_argument('--net_type', default='3conv3fc_1d', type=str, help='model')
    parser.add_argument('--dataset', default='UCI', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    if cfg.record_mean_var:
        mean_var_dir = f"checkpoints/{args.dataset}/bayesian/{args.net_type}/"
        cfg.mean_var_dir = mean_var_dir
        if not os.path.exists(mean_var_dir):
            os.makedirs(mean_var_dir, exist_ok=True)
        for file in os.listdir(mean_var_dir):
            os.remove(mean_var_dir + file)

    run(args.dataset, args.net_type,train=True)
