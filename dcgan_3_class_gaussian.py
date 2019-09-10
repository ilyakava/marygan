# -*- coding: utf-8 -*-

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import visdom

import scipy.misc

import pdb

# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

vis = visdom.Visdom()
last_itr_visuals = []


# Root directory for dataset
legend = ['fake', 'data1', 'data2']
# the smaller dataset should come first.
#dataroot = '/scratch0/ilya/locDoc/data/oxford-flowers'
dataroot = '/scratch0/ilya/locDoc/data/celeba_partitions/male_close'
# dataroot = '/scratch0/ilya/locDoc/data/StackGAN/Caltech-UCSD-Birds-200-2011/CUB_200_2011'
# dataroot2 = '/scratch0/ilya/locDoc/data/celeba'
dataroot2 = '/scratch0/ilya/locDoc/data/celeba_partitions/female_close'
# dataroot = '/scratch0/ilya/locDoc/data/mnist-M/mnist_m'
# dataroot2 = '/scratch0/ilya/locDoc/data/svhn'
# legend = ['fake', 'mu=-1', 'mu=+1']


outdata_path = '/scratch0/ilya/locDoc/MaryGAN/experiments/male_and_female_close4'

# Number of workers for dataloader
workers = 4

# Batch size during training
batch_size = 256

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
# image_size = 64 # not used
visdom_update_itrs = 1000

# Number of channels in the training images. For color images this is 3
nc = 2

# Size of z latent vector (i.e. size of generator input)
nz = 2

# Size of feature maps in generator
ngf = 512

# Size of feature maps in discriminator
ndf = 512

# Number of training epochs
num_epochs = 500

# Learning rate for optimizers 
# lr = 0.0001 # now
lr = 5e-5 # improved_wgan_training

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

mus = [(2,2), (2,4), (4,2), (4,4)]
# 9 gaussians
# mus = [(2,2), (2,4), (2,6), (4,2), (4,4), (4,6), (6,2), (6,4), (6,6)]
x_range = (0,8)
y_range = (0,8)
var = 1/2.0

# 2 gaussians
# mus = [(2,1), (2,3)]
# x_range = (0,4)
# y_range = (-1,5)

n_classes = len(mus)+1 # including fake

critic_iters = 1
labeled_iters = 1

# set to 0 to not use, 0.01 in improved_wgan_training github
clip_weights_value = 0.01

######################################################################

# Create a isotropic dataset
n_examples = 50000

dataloader = []
datalabels = []

for i, mu in enumerate(mus):
    dataloader.append(np.concatenate([np.random.normal(mu[0], var, (n_examples,1)), np.random.normal(mu[1], var, (n_examples,1))], axis=1))
    datalabels.append(i*np.ones(n_examples))

dataloader = np.concatenate(dataloader, 0)
datalabels = np.concatenate(datalabels, 0)

# shuffle
perm = np.random.permutation(dataloader.shape[0])
dataloader = dataloader[perm]
datalabels = datalabels[perm]

dataloader = torch.tensor(dataloader, dtype=torch.float)
datalabels = torch.tensor(datalabels, dtype=torch.long)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

######################################################################
# Implementation
# --------------

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 2.0 / ngf)

class Generator(nn.Module):
    # https://github.com/igul222/improved_wgan_training/blob/master/gan_toy.py
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z
            nn.Linear( nz, ngf),
            nn.ReLU(True),
            # 
            nn.Linear( ngf, ngf),
            nn.ReLU(True),
            # 
            nn.Linear( ngf, ngf),
            nn.ReLU(True),
            # 
            nn.Linear( ngf, nc),
            #
        )

    def forward(self, input):
        return self.main(input)

netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

print(netG)

class Discriminator(nn.Module):
    # https://github.com/igul222/improved_wgan_training/blob/master/gan_toy.py
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.net = nn.Sequential(
            # input is (nc)
            nn.Linear(nc, ndf),
            nn.ReLU(inplace=True),
            # 
            nn.Linear(ndf, ndf),
            nn.ReLU(inplace=True),
            #
            nn.Linear(ndf, ndf),
            nn.ReLU(inplace=True),
            #
            nn.Linear(ndf, n_classes)
        )
        self.softmax = nn.Softmax()

    def forward(self, input, probabilities=True):
        if probabilities:
            return self.softmax(self.net(input))
        else:
            return self.net(input)

netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

print(netD)


######################################################################
# Loss Functions and Optimizers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bce = nn.BCELoss()
nll = nn.NLLLoss()
def criterion(x,y):
    return torch.mean(y*x + (1-y)*(-x))
Vbce = nn.BCELoss(reduction='none')

def myloss(X, Y1hot, Ylabel, loss_type='nll'):
    """For trying multiple losses.
        Y1hot: target_1hot, target_lab

        wgan should take raw ouput of network, no softmax
    """
    # NLL case takes indices
    if loss_type == 'nll':
        return nll(torch.log(X), Ylabel)
    elif loss_type == 'wasserstein':
        inputs = X.gather(1,Ylabel.unsqueeze(-1))
        labels = Y1hot.gather(1,Ylabel.unsqueeze(-1))
        return torch.mean(inputs*labels) - torch.mean(inputs*(1-labels))
    elif loss_type == 'hinge':
        inputs = X.gather(1,Ylabel.unsqueeze(-1))
        labels = Y1hot.gather(1,Ylabel.unsqueeze(-1))
        return torch.mean(F.relu(1 + inputs*labels)) + torch.mean(F.relu(1 - inputs*(1-labels)))
    else:
        raise NotImplementedError('Loss type: %s' % loss_type)

    # older BCE case
    # return bce(X, Y1hot)

def mylossV(X,Y1hot):
    """
    Before -1 * max, can be log prob, bce prob, or hinge
    For Vbce: -1 * max makes bad gradients, need min
    """
    # return Vbce(X, Y1hot)
    return torch.log(X)


# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Adam
# optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
# RMS like in like in improved_wgan_training without grad penalty
optimizerD = torch.optim.RMSprop(netD.parameters(), lr=lr)
optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lr)

######################################################################
# Training
# ~~~~~~~~

def decimate_ts(ts, ds):
    win = np.ones(ds) / float(ds)
    return np.convolve(ts, win, mode='same')[::ds]

def decimate(y, ds):
    if ds > 1:
        if isinstance(y[0], list):
            num_ts = len(y[0])
            newy_transpose = []
            for i in range(num_ts):
                ts = [yi[i] for yi in y]
                newy_transpose.append(decimate_ts(ts, ds))
            return [list(x) for x in zip(*newy_transpose)]
        else:
            return decimate_ts(y, ds)
    else:
        return y

# Lists to keep track of progress
img_list = []
losses = []
real_D = []
fake_D = []
fake_D_gen = []
perf = []
perftime = []
y_coord_hist = []
waterfall_outf = '/scratch0/ilya/locDoc/MaryGAN/experiments/waterfall.npy'
hist2d_outf = '/scratch0/ilya/locDoc/MaryGAN/experiments/hist2d.npy'
H_hist = []
iters = 0

print("Starting Training Loop...")
start = time.time()
K = n_classes - 1

def labeled_batch_generator():
    while True:
        for i in range(0,dataloader.shape[0],batch_size):
            data = dataloader[i:i+batch_size,:]
            label = datalabels[i:i+batch_size]
            yield (data, label)

labeled_batch = labeled_batch_generator()
while True:
    for ci in range(critic_iters):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batches
        netD.zero_grad()
        
        errD_real = 0
        for li in range(labeled_iters):
            data, label = next(labeled_batch)
            real_cpu = data.to(device)
            output = netD(real_cpu,probabilities=False).squeeze()
            lab_labels_1hot = torch.zeros(output.shape, dtype=torch.float).scatter_(1, label.unsqueeze(-1), 1)
            label = label.to(device)
            lab_labels_1hot = lab_labels_1hot.to(device)
            label = K*torch.ones((output.shape[0],),dtype=torch.long,device=device)
            d_g1 = myloss(output, lab_labels_1hot, label, loss_type='hinge')
            errD_real += d_g1
        errD_real /= labeled_iters

        # Calculate gradients for D in backward pass
        errD_real.backward()
        if ci == (critic_iters - 1):
            real_D.append(torch.mean(output, dim=0).tolist())

        ## Train with all-fake batch
        noise = torch.randn(batch_size, nz, device=device)
        fake = netG(noise)
        output = netD(fake.detach(),probabilities=False).squeeze()
        label = K*torch.ones((output.shape[0],),dtype=torch.long,device=device)
        gen_labels_1hot = torch.zeros(output.shape, device=device).scatter_(1, label.unsqueeze(-1), 1)
        d_g0 = myloss(output, gen_labels_1hot, label, loss_type='hinge')
        errD_fake = d_g0
        
        # Calculate the gradients for this batch
        errD_fake.backward()
        if ci == (critic_iters - 1):
            fake_D.append(torch.mean(output, dim=0).tolist())
        
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        if clip_weights_value:
            for param in netD.parameters():
                param.data.clamp_(-clip_weights_value, clip_weights_value)

    if critic_iters > 1:
        noise = torch.randn(b_size, nz, device=device)
        fake = netG(noise)


    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    output = netD(fake,probabilities=False).squeeze()
    # fake labels are real for generator cost
    unl_labels_1hot = torch.ones(output.shape, device=device).scatter_(1, K*torch.ones((output.shape[0],1),dtype=torch.long,device=device), 0)
    # fake_loss = myloss(output[:,K], unl_labels_1hot[:,K])
    Vtrue_loss = mylossV(output[:,:K], unl_labels_1hot[:,:K])
    true_loss = -torch.mean(Vtrue_loss.max(1)[0])
    label = K*torch.ones((output.shape[0],),dtype=torch.long,device=device)
    true_loss = myloss(output, unl_labels_1hot, label, loss_type='hinge')
    
    # label = torch.from_numpy(np.random.randint(0, K, batch_size)).to(device)
    # alt_true_loss = nll(torch.log(output[:,:K]), label)
    
    errG = true_loss
    errG.backward()
    optimizerG.step()

    # errG = d0g0_g# d1g0_g + d2g0_g
    # Calculate gradients for G
    # if ...
    fake_D_gen.append(torch.mean(output, dim=0).tolist())
    # Update G
    
    if iters % 10 == 9:
        now = time.time()
        perf.append(10 / (now - start))
        start = now
        perftime.append(iters)
    # Output training stats
    if iters % 50 == 0:
        print('[%06d]\tLoss_D: %.4f\tLoss_G: %.4f'
              % (iters, errD.item(), errG.item()))
    
    # Save Losses for plotting later
    losses.append([errG.item(), errD.item()])


    # Check how the generator is doing by saving G's output on fixed_noise
    if (iters % visdom_update_itrs == 0):
        while len(last_itr_visuals) > 0:
            visual = last_itr_visuals.pop()
            vis.close(visual)

        with torch.no_grad():
            new_noise = torch.randn(50000, nz, device=device)
            moving_fake = netG(new_noise).detach().cpu()
            fixed_fake = netG(fixed_noise).detach().cpu()
        
        # scatterplot of moving fakes
        # plotX = np.concatenate([moving_fake.numpy()[:500], dataloader.numpy()[:64], dataloader2.numpy()[:64]])
        # plotY = np.concatenate([np.ones(500), 2*np.ones(64), 3*np.ones(64)])
        # last_itr_visuals.append(vis.scatter(plotX, plotY, opts={'legend': legend, 'title': 'Moving Fakes [epoch][itr]: [%d/%d][%d/%d]' % (epoch, num_epochs, i, len(dataloader)) }))
        
        # scatterplot of fixed fakes
        # the order of the legend follows the order of the label ordinals
        classes_legend = ['data_%d' % (i+1) for i in range(K)] + ['fake']
        plotX = np.concatenate([fixed_fake.numpy()[:64], dataloader[:128].numpy()])
        plotY = np.concatenate([(K+1)*np.ones(64), 1+datalabels[:128].numpy()])
        last_itr_visuals.append(vis.scatter(plotX, plotY, opts={'legend': classes_legend, 'title': 'Fixed Fakes [%d]:' % (iters) }))

        # histograms
        nxb = int((x_range[1] - x_range[0]) * 5)
        nyb = int((y_range[1] - y_range[0]) * 5)

        # histogram of x
        # x_coord, x_bins = np.histogram(moving_fake.numpy()[:,0], 2*nxb, x_range)
        # last_itr_visuals.append(vis.bar(x_coord, x_bins[:-1], opts={'title': 'Moving Fakes, x coord' }))
        # histogram of y
        y_coord, y_bins = np.histogram(moving_fake.numpy()[:,1], 2*nyb, y_range)
        last_itr_visuals.append(vis.bar(y_coord, y_bins[:-1], opts={'title': 'Moving Fakes, y coord' }))
        y_coord_hist.append(y_coord)
        np.save(waterfall_outf, y_coord_hist)

        # 2d histogram
        scalef = 1+(200 // np.min([nxb,nyb]))
        H, xedges, yedges = np.histogram2d(moving_fake.numpy()[:,0], moving_fake.numpy()[:,1], (nxb,nyb), [x_range, y_range], normed=True)
        Hlarge = scipy.misc.imresize(H, scalef*np.array(H.shape))
        last_itr_visuals.append(vis.image(np.flipud(Hlarge.T), opts={'title': 'Moving Fakes Heatmap'}))
        H_hist.append(H)
        np.save(hist2d_outf, H_hist)

        # fake_grid = vutils.make_grid(fixed_fake, padding=2, normalize=True)
        # moving_fake_grid = vutils.make_grid(moving_fake, padding=2, normalize=True)

        # last_itr_visuals.append(vis.image(fake_grid, opts={'title': 'Fixed Fakes [epoch][itr]: [%d/%d][%d/%d]' % (epoch, num_epochs, i, len(dataloader)) }))
        # last_itr_visuals.append(vis.image(moving_fake_grid, opts={'title': 'Moving Fakes [epoch][itr]: [%d/%d][%d/%d]' % (epoch, num_epochs, i, len(dataloader)) }))
        # save an image            
        # scipy.misc.imsave('%s/%06d.png' % (outdata_path, int(iters / 100)),
        #     np.moveaxis(fixed_fake[random.randint(0, fixed_fake.shape[0]-1),:,:,:].numpy(),0,-1)) 
        
        # plot some lines
        max_line_samples = 200
        ds = max(1,len(real_D) // (max_line_samples+1))

        # losses
        last_itr_visuals.append(vis.line(decimate(losses,ds), list(range(0,iters+1,ds)), opts={'legend': ['errG', 'errD'], 'title': 'Network Losses'}))

        # network outputs
        output_legend = ['output_%d' % i for i in range(n_classes)]
        last_itr_visuals.append(vis.line(decimate(real_D,ds), list(range(0,iters+1,ds)), opts={'legend': output_legend, 'title': 'Real classification'}))
        last_itr_visuals.append(vis.line(decimate(fake_D,ds), list(range(0,iters+1,ds)), opts={'legend': output_legend, 'title': 'Fake classification, D step'}))
        last_itr_visuals.append(vis.line(decimate(fake_D_gen,ds), list(range(0,iters+1,ds)), opts={'legend': output_legend, 'title': 'Fake classification, G step'}))

        # itrs per second
        # if perf:
        #     ds = max(1,len(perf) // (max_line_samples+1))
        #     last_itr_visuals.append(vis.line(decimate(perf, ds), perftime[::ds], opts={'title': 'iters per second'}))
        
    iters += 1
