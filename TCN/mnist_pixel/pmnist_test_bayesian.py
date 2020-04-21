import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append("../../")
from TCN.mnist_pixel.utils import data_generator
from TCN.mnist_pixel.model_bayesian import BBBTCN
import numpy as np
import argparse
from mnist_pixel.utils import logmeanexp
from mnist_pixel.metrics import ELBO, acc

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

root = './data/mnist'
batch_size = args.batch_size
n_classes = 10
input_channels = 1
seq_length = int(784 / input_channels)
epochs = args.epochs
steps = 0

print(args)
train_loader, test_loader = data_generator(root, batch_size)

permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
model = BBBTCN(n_classes, input_channels, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)

if args.cuda:
    model.cuda()
    permute = permute.cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def train(ep):
    num_ens = 1
    global steps
    train_loss = 0
    train_likihood_loss = 0
    train_kl_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)
        if args.permute:
            data = data[:, :, permute]
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        kl = 0.0
        outputs = torch.zeros(data.shape[0], n_classes, num_ens).to('cuda')
        for j in range(num_ens):
            net_out, _kl = model(data)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)
        # output = model(data)
        # loss = F.nll_loss(output, target)
        kl = kl / num_ens
        log_outputs = logmeanexp(outputs, dim=2)
        criterion = ELBO(len(data)).to('cuda')
        likihood_loss, kl_loss = criterion(log_outputs, target, kl)
        loss = likihood_loss + kl_loss
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss
        train_likihood_loss += likihood_loss
        train_kl_loss += kl_loss
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlikihood_loss:{:.6f}\tkl_loss:{:.6f}\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                train_likihood_loss.item()/args.log_interval,
                train_kl_loss.item() / args.log_interval,
                train_loss.item()/args.log_interval, steps))
            train_loss = 0
            train_likihood_loss = 0
            train_kl_loss = 0


def test():
    num_ens = 1
    model.eval()
    test_loss = 0
    test_likihood_loss = 0
    test_kl_loss = 0
    correct = 0
    accs = []
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            data, target = Variable(data, volatile=True), Variable(target)
            outputs = torch.zeros(data.shape[0], n_classes, num_ens).to('cuda')
            kl = 0.0
            for j in range(num_ens):
                net_out, _kl = model(data)
                kl += _kl
                outputs[:, :, j] = F.log_softmax(net_out, dim=1)

            log_outputs = logmeanexp(outputs, dim=2)
            criterion = ELBO(len(data)).to('cuda')
            likihood_loss, kl_loss = criterion(log_outputs, target, kl)
            test_likihood_loss += likihood_loss
            test_kl_loss += kl_loss
            loss = likihood_loss + kl_loss
            test_loss += loss
            accs.append(acc(log_outputs, target))
            # pred = log_outputs.data.max(1, keepdim=True)[1]
            # correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        test_likihood_loss /= len(test_loader.dataset)
        test_kl_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, likihood_loss: {:.4f}, kl_loss: {:.4f}, Accuracy: {:.4f}%\n'.format(
            test_loss, test_likihood_loss, test_kl_loss, np.mean(accs)))
        return test_loss


if __name__ == "__main__":
    for epoch in range(1, epochs+1):
        train(epoch)
        test()
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr