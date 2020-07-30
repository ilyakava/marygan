# -*- coding: utf-8 -*-
"""
Example usage:

# Improved GAN
CUDA_VISIBLE_DEVICES=1 python main.py --d_loss nll --g_loss feature_matching
# M-ary GAN
CUDA_VISIBLE_DEVICES=1 python main.py --d_loss nll --g_loss positive_log_likelihood
# dynamic activation maximization (Mode GAN)
CUDA_VISIBLE_DEVICES=1 python main.py --d_loss nll --g_loss activation_maximization
# Complement GAN
CUDA_VISIBLE_DEVICES=1 python main.py --d_loss nll --g_loss crammer_singer_complement --g_loss_aux confuse --g_loss_aux_weight 0.5
"""

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


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--d_lr', default=0.001, type=float)
parser.add_argument('--g_lr', default=0.001, type=float)
parser.add_argument("--d_loss", help="nll | activation_maximization | activation_minimization", default="nll")
parser.add_argument("--g_loss", help="see d_loss", default="positive_log_likelihood")
parser.add_argument("--g_loss_aux", help="see d_loss", default=None)
parser.add_argument('--g_loss_aux_weight', default=0.0, type=float)

args = parser.parse_args()

losses_on_logits = ['crammer_singer', 'crammer_singer_complement', 'confuse']
losses_on_features = ['feature_matching', 'feature_matching_l1']

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

# lr = 5e-5 # improved_wgan_training

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# mus = [(2,2), (2,4), (4,2), (4,4)]
# 9 gaussians
# mus = [(2,2), (2,4), (2,6), (4,2), (4,4), (4,6), (6,2), (6,4), (6,6)]
x_range = (0,8)
y_range = (0,8)
# var = 1/2.0

# circle of gaussians
K = 4 #10
xs = 2*np.cos(np.linspace(0,2*np.pi, K,endpoint=False)) + 4
ys = 2*np.sin(np.linspace(0,2*np.pi, K,endpoint=False)) + 4
var_numer = 8 # 2
variances = [(var_numer/8.0,var_numer/4.0)] * K
mus = list(zip(xs,ys))

n_classes = K+1 # including fake

critic_iters = 1
labeled_iters = 1

# set to 0 to not use, 0.01 in improved_wgan_training github
clip_weights_value = 0.1

######################################################################

# Create a isotropic dataset
n_examples = 10000

dataloader = []
datalabels = []

for i in range(K-1): # skip a gaussian here
    # specific to circle
    class_data = np.random.normal(0, variances[i], (n_examples,2))
    t = 2*np.pi*i/float(K)
    rot_mat = np.array([ [np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)] ])
    rotated_data = np.dot(rot_mat, class_data.T).T
    dataloader.append(rotated_data + np.array(mus[i]))
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
init_weight_var = 2.0 / ngf # he initialization
# init_weight_var = 0.02

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, init_weight_var)

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
        self.features = nn.Sequential(
            # input is (nc)
            nn.Linear(nc, ndf),
            nn.ReLU(inplace=True),
            # 
            nn.Linear(ndf, ndf),
            nn.ReLU(inplace=True),
            #
            nn.Linear(ndf, ndf),
            nn.ReLU(inplace=True)
        )
        self.logits = nn.Linear(ndf, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, input, probabilities=True, features=False):
        assert int(probabilities) + int(features) < 2, 'Cannot ask for both probabilities and features'
        if probabilities:
            return self.softmax(self.logits(self.features(input)))
        elif features:
            return self.features(input)
        else:
            return self.logits(self.features(input))

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

nll = nn.NLLLoss() # cross entropy but assumes log was already taken

def myloss(X=None, Ylabel=None, Xfeat=None, Yfeat=None, loss_type='nll'):
    """For trying multiple losses.
    Args:
        X: preds, could be probabilities or logits
        Ylabel: scalar labels
        
        Y1hot: target_1hot, target_lab

        wgan should take raw ouput of network, no softmax
    """
    # NLL case takes indices
    if loss_type == 'nll_builtin':
        return nll(torch.log(X), Ylabel)
    elif loss_type == 'nll':
        target = X.gather(1,Ylabel.unsqueeze(-1))
        return torch.mean(-torch.log(target))
    elif loss_type == 'feature_matching':
        return torch.mean((torch.mean(Xfeat, dim=0) - torch.mean(Yfeat, dim=0))**2)
    elif loss_type == 'feature_matching_l1':
        return torch.mean(torch.abs(torch.mean(Xfeat, dim=0) - torch.mean(Yfeat, dim=0)))
    elif loss_type == 'positive_log_likelihood':
        # go away from Ylabel
        target = X.gather(1,Ylabel.unsqueeze(-1))
        return torch.mean(torch.log(target))
    elif loss_type == 'activation_maximization':
        # Ylabel acts as 'target' to avoid
        mask = torch.ones_like(X)
        mask.scatter_(1, Ylabel.unsqueeze(-1), 0)
        wrongs = torch.masked_select(X,mask.byte()).reshape(X.shape[0],K)
        max_wrong, _ = wrongs.max(1)
        return torch.mean(-torch.log(max_wrong))
    elif loss_type == 'activation_minimization':
        # Ylabel acts as 'target' to avoid
        mask = torch.ones_like(X)
        mask.scatter_(1, Ylabel.unsqueeze(-1), 0)
        wrongs = torch.masked_select(X,mask.byte()).reshape(X.shape[0],K)
        min_wrong, _ = wrongs.min(1)
        return torch.mean(-torch.log(min_wrong))
    elif loss_type == 'confuse':
        confuse_margin = 0.1 #  0.01
        
        mask = torch.ones_like(X)
        mask.scatter_(1, Ylabel.unsqueeze(-1), 0)
        wrongs = torch.masked_select(X,mask.byte()).reshape(X.shape[0],K)
        max_wrong, max_Ylabel = wrongs.max(1)

        mask.scatter_(1, max_Ylabel.unsqueeze(-1), 0)
        wrongs2 = torch.masked_select(X,mask.byte()).reshape(X.shape[0],K-1)
        runnerup_wrong, _ = wrongs2.max(1)
        # make a step towards the margin if it is far from the margin
        return torch.mean(F.relu(-confuse_margin + max_wrong - runnerup_wrong))
    # elif loss_type == 'wasserstein':
    #     inputs = X.gather(1,Ylabel.unsqueeze(-1))
    #     labels = Y1hot.gather(1,Ylabel.unsqueeze(-1))
    #     return torch.mean(inputs*labels) - torch.mean(inputs*(1-labels))
    # elif loss_type == 'hinge':
    #     inputs = X.gather(1,Ylabel.unsqueeze(-1))
    #     labels = Y1hot.gather(1,Ylabel.unsqueeze(-1))
    #     return torch.mean(F.relu(1 + inputs*labels)) + torch.mean(F.relu(1 - inputs*(1-labels)))
    elif loss_type == 'crammer_singer':
        mask = torch.ones_like(X)
        mask.scatter_(1, Ylabel.unsqueeze(-1), 0)
        wrongs = torch.masked_select(X,mask.byte()).reshape(X.shape[0],K)
        max_wrong, _ = wrongs.max(1)
        max_wrong = max_wrong.unsqueeze(-1)
        target = X.gather(1,Ylabel.unsqueeze(-1))
        return torch.mean(F.relu(1 + max_wrong - target))
    elif loss_type == 'crammer_singer_complement':
        mask = torch.ones_like(X)
        mask.scatter_(1, Ylabel.unsqueeze(-1), 0)
        wrongs = torch.masked_select(X,mask.byte()).reshape(X.shape[0],K)
        max_wrong, _ = wrongs.max(1)
        max_wrong = max_wrong.unsqueeze(-1)
        target = X.gather(1,Ylabel.unsqueeze(-1))
        return torch.mean(F.relu(1 - max_wrong + target))
    else:
        raise NotImplementedError('Loss type: %s' % loss_type)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, device=device)

# Adam
# optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
# RMS like in like in improved_wgan_training without grad penalty
optimizerD = torch.optim.RMSprop(netD.parameters(), lr=args.d_lr)
optimizerG = torch.optim.RMSprop(netG.parameters(), lr=args.g_lr)

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
        # make sure the last leftover batch of with wrong size is skipped
        for j in range(args.batch_size,dataloader.shape[0],args.batch_size):
            i = j - args.batch_size
            data = dataloader[i:i+args.batch_size,:]
            label = datalabels[i:i+args.batch_size]
            yield (data, label)

labeled_batch = labeled_batch_generator()
while True:
    for ci in range(critic_iters):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        gd_kwargs = {'probabilities': True, 'features': False}
        if args.d_loss in losses_on_logits:
            gd_kwargs = {'probabilities': False, 'features': False}
            
        ## Train with all-real batches
        netD.zero_grad()
        probabilities = True
        
        errD_real = 0
        for li in range(labeled_iters):
            data, label = next(labeled_batch)
            real_cpu = data.to(device)
            output = netD(real_cpu,** gd_kwargs).squeeze()
            label = label.to(device)

            errD_main = myloss(output, label, loss_type=args.d_loss)

            errD_real += errD_main
            
        errD_real /= labeled_iters

        # Calculate gradients for D in backward pass
        errD_real.backward()
        if ci == (critic_iters - 1):
            real_D.append(torch.mean(output, dim=0).tolist())

        ## Train with all-fake batch
        noise = torch.randn(args.batch_size, nz, device=device)
        fake = netG(noise)
        output = netD(fake.detach(), **gd_kwargs).squeeze()
        label = K*torch.ones((output.shape[0],),dtype=torch.long,device=device)
        
        d_g0 = myloss(output, label, loss_type=args.d_loss)
        errD_fake = d_g0
        
        # Calculate the gradients for this batch
        errD_fake.backward()
        if ci == (critic_iters - 1):
            fake_D.append(torch.mean(output, dim=0).tolist())
        
        # Add the gradients from the all-real and all-fake batches
        errD = (errD_real + errD_fake) / 2.0
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
    gd_kwargs = {'probabilities': True, 'features': False}
    if args.g_loss in losses_on_features:
        gd_kwargs = {'probabilities': False, 'features': True}
    if args.g_loss in losses_on_logits:
        gd_kwargs = {'probabilities': False, 'features': False}
    real_features = None
    fake_features = None
    output = None
    
    netG.zero_grad()
    if gd_kwargs['features']:
        fake_features = netD(fake,**gd_kwargs).squeeze()
        real_features = netD(real_cpu,**gd_kwargs).detach().squeeze()
    else:
        output = netD(fake,**gd_kwargs).squeeze()

    # unl_labels_1hot = torch.ones(output.shape, device=device).scatter_(1, K*torch.ones((output.shape[0],1),dtype=torch.long,device=device), 0)
    fake_class_idx = K*torch.ones((fake.shape[0],),dtype=torch.long,device=device)
    
    errG_main = myloss(X=output, Ylabel=fake_class_idx, Xfeat=fake_features, Yfeat=real_features, loss_type=args.g_loss)
    
    if args.g_loss_aux is not None:
        errG_aux = myloss(X=output, Ylabel=fake_class_idx, Xfeat=fake_features, Yfeat=real_features, loss_type=args.g_loss_aux)
    else: 
        errG_aux = 0.0
    
    errG = (1 - args.g_loss_aux_weight) * errG_main + args.g_loss_aux_weight * errG_aux
    
    # Calculate gradients for G
    errG.backward()
    # Update G
    optimizerG.step()

    # fake_D_gen.append(torch.mean(output, dim=0).tolist())
    

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

        # 2d histogram of generated
        scalef = 1+(200 // np.min([nxb,nyb]))
        H, xedges, yedges = np.histogram2d(moving_fake.numpy()[:,0], moving_fake.numpy()[:,1], (nxb,nyb), [x_range, y_range], normed=True)
        Hlarge = scipy.misc.imresize(H, scalef*np.array(H.shape))
        last_itr_visuals.append(vis.image(np.flipud(Hlarge.T), opts={'title': 'Moving Fakes Heatmap'}))
        H_hist.append(H)
        np.save(hist2d_outf, H_hist)

        # 2d histogram of real
        if iters == 0:
            scalef = 1+(200 // np.min([nxb,nyb]))
            H, xedges, yedges = np.histogram2d(dataloader.numpy()[:,0], dataloader.numpy()[:,1], (nxb,nyb), [x_range, y_range], normed=True)
            Hlarge = scipy.misc.imresize(H, scalef*np.array(H.shape))
            vis.image(np.flipud(Hlarge.T), opts={'title': 'Real Heatmap'})

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
        # last_itr_visuals.append(vis.line(decimate(fake_D_gen,ds), list(range(0,iters+1,ds)), opts={'legend': output_legend, 'title': 'Fake classification, G step'}))

        # itrs per second
        # if perf:
        #     ds = max(1,len(perf) // (max_line_samples+1))
        #     last_itr_visuals.append(vis.line(decimate(perf, ds), perftime[::ds], opts={'title': 'iters per second'}))
        
    iters += 1
