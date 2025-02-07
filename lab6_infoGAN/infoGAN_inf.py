
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=80, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
#parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--cuda', default='true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

T = transforms.Compose([transforms.Resize(64),transforms.ToTensor()])
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,transform=T),batch_size=opt.batchSize, shuffle=True)


ngpu = int(opt.ngpu)
nz = int(opt.nz)    #64
ngf = int(opt.ngf)  #64 generator
ndf = int(opt.ndf)  #64 discriminator
nc = 1

def to_var(x, volatile=False):
    return Variable(x, volatile=volatile)

def idx2onehot(idx, n):
    onehot = torch.zeros(idx.size(0), n).cuda()
    onehot.scatter_(1, idx.data, 1)
    onehot = to_var(onehot)
    return onehot

def _noise_sample(dis_c, noise, bs):
    idx = np.random.randint(10, size=bs)
    c = np.zeros((bs, 10))
    c[range(bs),idx] = 1.0

    dis_c.data.copy_(torch.Tensor(c))
    noise.data.uniform_(-1.0, 1.0)
    z = torch.cat([noise, dis_c], 1).view(-1, 64, 1, 1)
    return z, idx

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), #in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True
            nn.BatchNorm2d(ngf * 8, eps= 1e-05, momentum= 0.1, affine= True),
            nn.ReLU(inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4, eps= 1e-05, momentum= 0.1, affine= True),
            nn.ReLU(inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2, eps= 1e-05, momentum= 0.1, affine= True),
            nn.ReLU(inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf, eps= 1e-05, momentum= 0.1, affine= True),
            nn.ReLU(inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),#one channel
            nn.Tanh()#Probability
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output


netG = _netG(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2, eps= 1e-05, momentum= 0.1, affine= True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4, eps= 1e-05, momentum= 0.1, affine= True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8, eps= 1e-05, momentum= 0.1, affine= True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.d = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.q = nn.Sequential(    #classify
            # state size. (ndf*8) x 4 x 4
            nn.Linear(8192, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 10, bias=True)
        )


    def forward(self, input,isQ = False): #if isQ then classify
        output = self.main(input)

        if isQ:
            output = output.view(output.size(0), -1)
            output = self.q(output)
        else:
            output = self.d(output)

        return output.view(-1, 1).squeeze(1)


netD = _netD(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)


criterion = nn.BCELoss() #ln=−wn[yn⋅logxn+(1−yn)⋅log(1−xn)]
criterionQ = nn.CrossEntropyLoss()

input = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterionQ.cuda()
    input, label = input.cuda(), label.cuda()

# setup optimizer
optimizerD = optim.Adam([{'params':netD.main.parameters()}, {'params':netD.d.parameters()}], lr=2e-4, betas=(opt.beta1, 0.99))
optimizerG = optim.Adam([{'params':netG.parameters()}, {'params':netD.q.parameters()}], lr=1e-3, betas=(opt.beta1, 0.99))


dis_c = torch.FloatTensor(opt.batchSize, 10).cuda()
noise = torch.FloatTensor(opt.batchSize, 54).cuda()

dis_c = Variable(dis_c)
noise = Variable(noise)

idx = np.arange(10).repeat(10)
one_hot = np.zeros((100, 10))
one_hot[range(100), idx] = 1
fix_noise = torch.FloatTensor(opt.batchSize, 54).cuda()
fix_noise = Variable(fix_noise)

loss_listG= []
loss_listD= []
loss_listQ= []
D_x_list= []
D_G1_list= []
D_G2_list= []


for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # train with real
        netD.zero_grad() #initial
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        dis_c.data.resize_(batch_size, 10)
        noise.data.resize_(batch_size, 54)
        real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)
        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()
        
        # train with fake
        #noise.resize_(batch_size, 54, 1, 1).normal_(0, 1)
        #noisev = Variable(noise)
        z, idx = _noise_sample(dis_c, noise, batch_size)
        #print(z)
        fake = netG(z)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        
        errD = errD_real + errD_fake
        optimizerD.step() #optimize disc.
        #print(errD)
        # (2) Update G network: maximize log(D(G(z)))
        
        netG.zero_grad() #initial
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labelv)
        ##train classifier
        q_logits =netD(fake,isQ=True) #classify
        class_ = torch.LongTensor(idx).cuda()#real answer
        target = Variable(class_)
        q_logits.data.resize_(q_logits.size(0)//10,10)
        errQ = criterionQ(q_logits,target)
        G_loss = errG + errQ
		##print(G_loss)
        G_loss.backward()
        D_G_z2 = G_loss.data.mean()
        optimizerG.step()
        fake = netG(z)
        output = netD(fake.detach())
        D_G_z2 = output.data.mean()
        
        #####print(D_G_z2)
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_Q: %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], errQ.data[0]))
        
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True,nrow=8)
            c = to_var(torch.arange(0,10).long().view(-1,1)).cuda()
            c = idx2onehot(c, n=10)
            tmp = c
            for i in range(1,10):
                tmp = torch.cat((tmp, c), dim=-1)
            c_s = torch.transpose(tmp, 0, 1)
            sample = to_var(torch.randn(54, 10, 1).normal_()).cuda()
            sample.data.uniform_(-1.0, 1.0)
            tmp = sample
            for i in range(1,10):
                tmp = torch.cat((tmp, sample), dim=-1)
            s_s = torch.transpose(tmp, 1, 2)
            s_s.data.resize_(54, 100)
            s_s = torch.transpose(s_s, 0, 1)
            sample = Variable(torch.randn(100, 20))
            #noise size = 20, condition code size = 10
            sample = torch.cat((s_s, c_s), dim=-1)
            sample.data.resize_(100,64,1,1)
            fake = netG(sample)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True ,nrow=10)
            loss_listD.append(errD.data[0])
            loss_listG.append(errG.data[0])
            loss_listQ.append(errQ.data[0])
            D_x_list.append(D_x)
            D_G1_list.append(D_G_z1)
            D_G2_list.append(D_G_z2)
    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
plt.title('Loss Curve')
plt.ylabel('loss')
plt.plot(np.arange(1,len(loss_listG)+1),loss_listD , color = 'red')
plt.plot(np.arange(1,len(loss_listD)+1),loss_listG , color = 'black')
plt.plot(np.arange(1,len(loss_listQ)+1),loss_listQ , color = 'yellow')
plt.legend(loc='upper right')
savefilename = 'infoGAN_training_loss.jpg'
plt.savefig(savefilename)
plt.close()
plt.clf()
plt.title('P')
plt.ylabel('Infogan/prob')
plt.plot(np.arange(1,len(D_x_list)+1),D_x_list , color = 'red')
plt.plot(np.arange(1,len(D_G1_list)+1),D_G1_list , color = 'black')
plt.plot(np.arange(1,len(D_G2_list)+1),D_G2_list , color = 'yellow')
plt.legend(loc='upper right')
savefilename = 'infoGAN_p.jpg'
plt.savefig(savefilename)
plt.close()

label = 0
netG.cuda()
#netG.load_state_dict(torch.load('resultsGAN/netG_epoch_79.pth'))
for i in range(5):
    onehot = np.zeros((1, 10))
    onehot[range(1),label] = 1.0
    onehot = Variable(torch.FloatTensor(onehot)).cuda()

    sample = Variable(torch.FloatTensor(1, 54).uniform_(-1.0, 1.0)).cuda()

    z = torch.cat([sample, onehot], 1).view(-1, 64, 1, 1)

    fake = netG(z)
    vutils.save_image(fake.data,'%s/GANoutput_%d.png' % (opt.outf,i),normalize=True ,nrow=10)