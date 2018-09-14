from __future__ import print_function
import os
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import sys
import csv
import numpy as np


if not os.path.exists('results'):
    os.makedirs('results')

if not os.path.exists('model_save'):
    os.makedirs('model_save')
fdir = 'model_save'
best_prec = 0

def save_checkpoint(para, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(para, filepath)
    if is_best:
        torch.save(para, os.path.join(fdir, 'model_best.pth.tar'))

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=5, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST_data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST_data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


best_loss=0

class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
   
        #enc
        self.fc1 = nn.Linear(784, 400, bias=True)
        self.fc21 = nn.Linear(400, 20, bias=True)
        self.fc22 = nn.Linear(400, 20, bias=True)

        #dec
        self.fc3 = nn.Linear(30, 392, bias=True)

        #use conv not linear
        
        #stride=(1, 1), padding=(1, 1), kernel_size=(3, 3)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(2, 11, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(11, 3, kernel_size=3, stride=1, padding=1)
		
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def encode(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(-1, 784)
        x = self.fc1(x)
        h1 = F.relu(x)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, c):
        # print('miku')
        # print(c.shape)
        # print('========================')
        # print(z.shape)
        z = torch.cat([z, c], 1)
        # print(z.shape)
        # when cat = 1, then it combine element to dim1, dim0 will not change
        # when cat = 0, oppisite
        # print(z.shape)
        
        z = self.fc3(z)
        z = F.relu(z)	
        z = z.view(-1, 2, 14, 14)
        z = self.conv3(z)
        # print(z.shape)
        z = F.relu(z)
        z = self.upsample(z)
        # print(z.shape)
        z = self.conv4(z)
        # print(z.shape)
        z = F.relu(z)
        z = self.conv2(z)
        # print(z.shape)
        return F.sigmoid(z) 

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

def to_var(x):
    x = Variable(x)
    x = x.cuda()
    return x

def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)


model = CVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    recon_x = recon_x.view(-1, 784)
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    MSE = F.mse_loss(recon_x, x.view(-1, 784), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD

yy = []
def train(epoch):
    global best_loss
    model.train()
    train_loss = 0
    print('Epoch\t Loss')
    for batch_idx, (data, labels) in enumerate(train_loader):

        data = data.to(device)
        # print(type(labels))
        
        labels = one_hot(labels, 10)
        # print(labels.shape)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, labels)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        # print(loss.item())
        yy.append(loss.item())
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
            #print('{0}\t {1}'.format(epoch, loss.item() / len(data)))
            
            is_best = loss.item()/len(data) < best_loss
            best_loss = min(loss.item()/len(data), best_loss)
            save_checkpoint(model.state_dict(), is_best, fdir)
            sys.stdout.flush()



    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    #print('Epoch\t Aver_loss')
    #print('{0}\t {1}'.format(epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            # print('miku=============='+str(i))
            # print(labels)
            labels = one_hot(labels, 10)
            # print(labels)
            recon_batch, mu, logvar = model(data, labels)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    #print('Test loss\t {:.4f}'.format(test_loss))


#print(best_loss) zeros
for epoch in range(1, args.epochs + 1):
    train(epoch)
    print('Finish train')
    test(epoch)
    with torch.no_grad():
        c = torch.eye(10, 10)
        # b = np.full((10, 10), 2)
        for i in range(0,9):
            c = torch.cat([c, torch.eye(10, 10)], 0)
        ###############################################################
        c.fill_(9) #where we control the number
        ###############################################################
        c = to_var(c)
        c = c.long()
        # print(c)
        c = one_hot(c, 10)
        print(c)
        # print(c.shape)
        z = to_var(torch.randn(100, 20))
        z_test = to_var(torch.zeros(100, 20))
        #sample = torch.randn(64, 20).to(device)
        # print(z)
        sample = model.decode(z, c).cpu()
        sample = sample.view(100, 1, 28, 28)
        save_image(sample[1:11],
                   'results/sample_' + str(epoch) + '.png',nrow=10)
                   
ffnn = 'CVAE_CURVE.csv' #use excel to output the loss
with open(ffnn, 'w') as csvFile:
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(yy)