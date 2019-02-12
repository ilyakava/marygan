# -*- coding: utf-8 -*-

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import time

import torch
import torch.nn as nn
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

######################################################################
# Inputs
# ------
# 
# Let’s define some inputs for the run:
# 
# -  **dataroot** - the path to the root of the dataset folder. We will
#    talk more about the dataset in the next section
# -  **workers** - the number of worker threads for loading the data with
#    the DataLoader
# -  **batch_size** - the batch size used in training. The DCGAN paper
#    uses a batch size of 128
# -  **image_size** - the spatial size of the images used for training.
#    This implementation defaults to 64x64. If another size is desired,
#    the structures of D and G must be changed. See
#    `here <https://github.com/pytorch/examples/issues/70>`__ for more
#    details
# -  **nc** - number of color channels in the input images. For color
#    images this is 3
# -  **nz** - length of latent vector
# -  **ngf** - relates to the depth of feature maps carried through the
#    generator
# -  **ndf** - sets the depth of feature maps propagated through the
#    discriminator
# -  **num_epochs** - number of training epochs to run. Training for
#    longer will probably lead to better results but will also take much
#    longer
# -  **lr** - learning rate for training. As described in the DCGAN paper,
#    this number should be 0.0002
# -  **beta1** - beta1 hyperparameter for Adam optimizers. As described in
#    paper, this number should be 0.5
# -  **ngpu** - number of GPUs available. If this is 0, code will run in
#    CPU mode. If this number is greater than 0 it will run on that number
#    of GPUs
# 

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
lr = 0.0001 # now

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 2

n_classes = 3

critic_iters = 5


######################################################################
# Data
# ----
#
# In this tutorial we will use the `Celeb-A Faces
# dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`__ which can
# be downloaded at the linked site, or in `Google
# Drive <https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg>`__.
# The dataset will download as a file named *img_align_celeba.zip*. Once
# downloaded, create a directory named *celeba* and extract the zip file
# into that directory. Then, set the *dataroot* input for this notebook to
# the *celeba* directory you just created. The resulting directory
# structure should be:
# 
# ::
# 
#    /path/to/celeba
#        -> img_align_celeba  
#            -> 188242.jpg
#            -> 173822.jpg
#            -> 284702.jpg
#            -> 537394.jpg
#               ...
# 
# This is an important step because we will be using the ImageFolder
# dataset class, which requires there to be subdirectories in the
# dataset’s root folder. Now, we can create the dataset, create the
# dataloader, set the device to run on, and finally visualize some of the
# training data.
# 

# We can use an image folder dataset the way we have it setup.
# Create a isotropic dataset
n_examples = 50000

dataloader = torch.tensor(np.random.normal(2, 1, nc*n_examples).reshape((n_examples,nc)),dtype=torch.float)
# dataloader2 = torch.tensor(np.random.normal(1, 0.1, nc*n_examples).reshape((n_examples,nc)),dtype=torch.float)
dataloader2 = torch.tensor(np.concatenate([np.random.normal(2, 1, (n_examples,1)), np.random.normal(4, 1, (n_examples,1))], axis=1),dtype=torch.float)

x_range = (0,4)
y_range = (0,6)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

######################################################################
# Implementation
# --------------
# 
# With our input parameters set and the dataset prepared, we can now get
# into the implementation. We will start with the weigth initialization
# strategy, then talk about the generator, discriminator, loss functions,
# and training loop in detail.
# 
# Weight Initialization
# ~~~~~~~~~~~~~~~~~~~~~
# 
# From the DCGAN paper, the authors specify that all model weights shall
# be randomly initialized from a Normal distribution with mean=0,
# stdev=0.2. The ``weights_init`` function takes an initialized model as
# input and reinitializes all convolutional, convolutional-transpose, and
# batch normalization layers to meet this criteria. This function is
# applied to the models immediately after initialization.
# 

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


######################################################################
# Generator
# ~~~~~~~~~
# 
# The generator, :math:`G`, is designed to map the latent space vector
# (:math:`z`) to data-space. Since our data are images, converting
# :math:`z` to data-space means ultimately creating a RGB image with the
# same size as the training images (i.e. 3x64x64). In practice, this is
# accomplished through a series of strided two dimensional convolutional
# transpose layers, each paired with a 2d batch norm layer and a relu
# activation. The output of the generator is fed through a tanh function
# to return it to the input data range of :math:`[-1,1]`. It is worth
# noting the existence of the batch norm functions after the
# conv-transpose layers, as this is a critical contribution of the DCGAN
# paper. These layers help with the flow of gradients during training. An
# image of the generator from the DCGAN paper is shown below.
#
# .. figure:: /_static/img/dcgan_generator.png
#    :alt: dcgan_generator
#
# Notice, the how the inputs we set in the input section (*nz*, *ngf*, and
# *nc*) influence the generator architecture in code. *nz* is the length
# of the z input vector, *ngf* relates to the size of the feature maps
# that are propagated through the generator, and *nc* is the number of
# channels in the output image (set to 3 for RGB images). Below is the
# code for the generator.
# 

# Generator Code

class Generator(nn.Module):
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
######################################################################
# Now, we can instantiate the generator and apply the ``weights_init``
# function. Check out the printed model to see how the generator object is
# structured.
# 

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)


######################################################################
# Discriminator
# ~~~~~~~~~~~~~
# 
# As mentioned, the discriminator, :math:`D`, is a binary classification
# network that takes an image as input and outputs a scalar probability
# that the input image is real (as opposed to fake). Here, :math:`D` takes
# a 3x64x64 input image, processes it through a series of Conv2d,
# BatchNorm2d, and LeakyReLU layers, and outputs the final probability
# through a Sigmoid activation function. This architecture can be extended
# with more layers if necessary for the problem, but there is significance
# to the use of the strided convolution, BatchNorm, and LeakyReLUs. The
# DCGAN paper mentions it is a good practice to use strided convolution
# rather than pooling to downsample because it lets the network learn its
# own pooling function. Also batch norm and leaky relu functions promote
# healthy gradient flow which is critical for the learning process of both
# :math:`G` and :math:`D`.
# 

#########################################################################
# Discriminator Code

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
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
            nn.Linear(ndf, n_classes),
            nn.Softmax()
        )

    def forward(self, input):
        return self.main(input)


######################################################################
# Now, as with the generator, we can create the discriminator, apply the
# ``weights_init`` function, and print the model’s structure.
# 

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)


######################################################################
# Loss Functions and Optimizers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# With :math:`D` and :math:`G` setup, we can specify how they learn
# through the loss functions and optimizers. We will use the Binary Cross
# Entropy loss
# (`BCELoss <https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss>`__)
# function which is defined in PyTorch as:
# 
# .. math:: \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad l_n = - \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]
# 
# Notice how this function provides the calculation of both log components
# in the objective function (i.e. :math:`log(D(x))` and
# :math:`log(1-D(G(z)))`). We can specify what part of the BCE equation to
# use with the :math:`y` input. This is accomplished in the training loop
# which is coming up soon, but it is important to understand how we can
# choose which component we wish to calculate just by changing :math:`y`
# (i.e. GT labels).
# 
# Next, we define our real label as 1 and the fake label as 0. These
# labels will be used when calculating the losses of :math:`D` and
# :math:`G`, and this is also the convention used in the original GAN
# paper. Finally, we set up two separate optimizers, one for :math:`D` and
# one for :math:`G`. As specified in the DCGAN paper, both are Adam
# optimizers with learning rate 0.0002 and Beta1 = 0.5. For keeping track
# of the generator’s learning progression, we will generate a fixed batch
# of latent vectors that are drawn from a Gaussian distribution
# (i.e. fixed_noise) . In the training loop, we will periodically input
# this fixed_noise into :math:`G`, and over the iterations we will see
# images form out of the noise.
# 

# Initialize BCELoss function
criterion = nn.BCELoss()
# def criterion(x,y):
#     return torch.mean(y*x + (1-y)*(-x))

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


######################################################################
# Training
# ~~~~~~~~
# 
# Finally, now that we have all of the parts of the GAN framework defined,
# we can train it. Be mindful that training GANs is somewhat of an art
# form, as incorrect hyperparameter settings lead to mode collapse with
# little explanation of what went wrong. Here, we will closely follow
# Algorithm 1 from Goodfellow’s paper, while abiding by some of the best
# practices shown in `ganhacks <https://github.com/soumith/ganhacks>`__.
# Namely, we will “construct different mini-batches for real and fake”
# images, and also adjust G’s objective function to maximize
# :math:`logD(G(z))`. Training is split up into two main parts. Part 1
# updates the Discriminator and Part 2 updates the Generator.
# 
# **Part 1 - Train the Discriminator**
# 
# Recall, the goal of training the discriminator is to maximize the
# probability of correctly classifying a given input as real or fake. In
# terms of Goodfellow, we wish to “update the discriminator by ascending
# its stochastic gradient”. Practically, we want to maximize
# :math:`log(D(x)) + log(1-D(G(z)))`. Due to the separate mini-batch
# suggestion from ganhacks, we will calculate this in two steps. First, we
# will construct a batch of real samples from the training set, forward
# pass through :math:`D`, calculate the loss (:math:`log(D(x))`), then
# calculate the gradients in a backward pass. Secondly, we will construct
# a batch of fake samples with the current generator, forward pass this
# batch through :math:`D`, calculate the loss (:math:`log(1-D(G(z)))`),
# and *accumulate* the gradients with a backward pass. Now, with the
# gradients accumulated from both the all-real and all-fake batches, we
# call a step of the Discriminator’s optimizer.
# 
# **Part 2 - Train the Generator**
# 
# As stated in the original paper, we want to train the Generator by
# minimizing :math:`log(1-D(G(z)))` in an effort to generate better fakes.
# As mentioned, this was shown by Goodfellow to not provide sufficient
# gradients, especially early in the learning process. As a fix, we
# instead wish to maximize :math:`log(D(G(z)))`. In the code we accomplish
# this by: classifying the Generator output from Part 1 with the
# Discriminator, computing G’s loss *using real labels as GT*, computing
# G’s gradients in a backward pass, and finally updating G’s parameters
# with an optimizer step. It may seem counter-intuitive to use the real
# labels as GT labels for the loss function, but this allows us to use the
# :math:`log(x)` part of the BCELoss (rather than the :math:`log(1-x)`
# part) which is exactly what we want.
# 
# Finally, we will do some statistic reporting and at the end of each
# epoch we will push our fixed_noise batch through the generator to
# visually track the progress of G’s training. The training statistics
# reported are:
# 
# -  **Loss_D** - discriminator loss calculated as the sum of losses for
#    the all real and all fake batches (:math:`log(D(x)) + log(D(G(z)))`).
# -  **Loss_G** - generator loss calculated as :math:`log(D(G(z)))`
# -  **D(x)** - the average output (across the batch) of the discriminator
#    for the all real batch. This should start close to 1 then
#    theoretically converge to 0.5 when G gets better. Think about why
#    this is.
# -  **D(G(z))** - average discriminator outputs for the all fake batch.
#    The first number is before D is updated and the second number is
#    after D is updated. These numbers should start near 0 and converge to
#    0.5 as G gets better. Think about why this is.
# 
# **Note:** This step might take a while, depending on how many epochs you
# run and if you removed some data from the dataset.
# 

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

# Training Loop

# Lists to keep track of progress
img_list = []
real_losses_detail = []
fake_losses_detail = []
losses = []
data1_D = []
data2_D = []
fake_D = []
fake_D_gen = []
perf = []
perftime = []
y_coord_hist = []
waterfall_outf = '/scratch0/ilya/locDoc/MaryGAN/experiments/waterfall3.npy'
iters = 0

print("Starting Training Loop...")
start = time.time()
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i in range(0,n_examples,batch_size):
        data = [dataloader[i:i+batch_size,:]]
        data2 = [dataloader2[i:i+batch_size,:]]
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batches
        # first dataset
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)

        label = torch.tensor([0,1,0], dtype=torch.float, device=device).repeat(b_size,1)
        # Forward pass real batch through D
        output = netD(real_cpu).squeeze()
        # Calculate loss on all-real batch
        # loss variable names will be decision given true
        d_g1 = criterion(output, label)
        d0g1 = criterion(output[:,0], label[:,0])
        d2g1 = criterion(output[:,2], label[:,2])
        errD_real = d_g1 / 2 # d0g1# + d2g1
        # Calculate gradients for D in backward pass
        errD_real.backward()
        # optimizerD.step()
        # netD.zero_grad()
        data1_D.append(torch.mean(output, dim=0).tolist())



        # second dataset 
        real_cpu2 = data2[0].to(device)
        b_size2 = real_cpu2.size(0)

        label = torch.tensor([0,0,1], dtype=torch.float, device=device).repeat(b_size2,1)
        # Forward pass real batch through D
        output = netD(real_cpu2).squeeze()
        # Calculate loss on all-real batch
        d_g2 = criterion(output, label)
        d0g2 = criterion(output[:,0], label[:,0])
        d1g2 = criterion(output[:,1], label[:,1])
        errD_real2 = d_g2 / 2# d0g2# + d1g2
        # Calculate gradients for D in backward pass
        errD_real2.backward()
        # optimizerD.step()
        # netD.zero_grad()
        data2_D.append(torch.mean(output, dim=0).tolist())

        # errD_real_all = errD_real2
        errD_real_all = (errD_real + errD_real2) / 2
        # errD_real_all.backward()


        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label = torch.tensor([1,0,0], dtype=torch.float, device=device).repeat(b_size,1)
        # Classify all fake batch with D
        output = netD(fake.detach()).squeeze()
        # Calculate D's loss on the all-fake batch
        d_g0 = criterion(output, label)
        d0g0 = criterion(output[:,0], label[:,0])
        d1g0 = criterion(output[:,1], label[:,1])
        d2g0 = criterion(output[:,2], label[:,2])
        errD_fake = d_g0#d0g0#d1g0 + d2g0
        # Calculate the gradients for this batch
        errD_fake.backward()
        fake_D.append(torch.mean(output, dim=0).tolist())
        # Add the gradients from the all-real and all-fake batches
        # errD = errD_real + errD_real2 + errD_fake
        errD = errD_real_all + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        # fake labels are real for generator cost
        label = torch.tensor([0,1,1], dtype=torch.float, device=device).repeat(b_size,1)
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).squeeze()
        # Calculate G's loss based on this output
        d_g0_g = criterion(output, label)
        d0g0_g = criterion(output[:,0], label[:,0])
        d1g0_g = criterion(output[:,1], label[:,1])
        d2g0_g = criterion(output[:,2], label[:,2])
        errG = d_g0_g#d0g0_g# d1g0_g + d2g0_g
        # errG = 2*d0g0_g
        # Calculate gradients for G
        errG.backward()
        fake_D_gen.append(torch.mean(output, dim=0).tolist())
        # Update G
        optimizerG.step()
        
        if iters % 10 == 9:
            now = time.time()
            perf.append(10 / (now - start))
            start = now
            perftime.append(iters)
        # Output training stats
        if i % 50 == 0:
            print('[%03d/%03d][%04d/%04d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item()))
        
        # Save Losses for plotting later
        losses.append([errG.item(), errD.item()])
        real_losses_detail.append([d0g1.item(), d2g1.item(), d0g2.item(), d1g2.item()])
        fake_losses_detail.append([d1g0.item(), d2g0.item(), d1g0_g.item(), d2g0_g.item(), d0g0_g.item()])


        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % visdom_update_itrs == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            while len(last_itr_visuals) > 0:
                visual = last_itr_visuals.pop()
                vis.close(visual)

            with torch.no_grad():
                new_noise = torch.randn(2500, nz, device=device)
                moving_fake = netG(new_noise).detach().cpu()
                fixed_fake = netG(fixed_noise).detach().cpu()
            
            # scatterplot of moving fakes
            # plotX = np.concatenate([moving_fake.numpy()[:500], dataloader.numpy()[:64], dataloader2.numpy()[:64]])
            # plotY = np.concatenate([np.ones(500), 2*np.ones(64), 3*np.ones(64)])
            # last_itr_visuals.append(vis.scatter(plotX, plotY, opts={'legend': legend, 'title': 'Moving Fakes [epoch][itr]: [%d/%d][%d/%d]' % (epoch, num_epochs, i, len(dataloader)) }))
            
            # scatterplot of fixed fakes
            plotX = np.concatenate([fixed_fake.numpy()[:64], dataloader.numpy()[:64], dataloader2.numpy()[:64]])
            plotY = np.concatenate([np.ones(64), 2*np.ones(64), 3*np.ones(64)])
            last_itr_visuals.append(vis.scatter(plotX, plotY, opts={'legend': legend, 'title': 'Fixed Fakes [epoch][itr]: [%d/%d][%d/%d]' % (epoch, num_epochs, i, len(dataloader)) }))

            # histogram of x
            # x_coord, x_bins = np.histogram(moving_fake.numpy()[:,0], 50, x_range)
            # last_itr_visuals.append(vis.bar(x_coord, x_bins[:-1], opts={'title': 'Moving Fakes, x coord' }))
            # histogram of y
            y_coord, y_bins = np.histogram(moving_fake.numpy()[:,1], 50, y_range)
            last_itr_visuals.append(vis.bar(y_coord, y_bins[:-1], opts={'title': 'Moving Fakes, y coord' }))
            y_coord_hist.append(y_coord)
            np.save(waterfall_outf, y_coord_hist)

            # 2d histogram
            H, xedges, yedges = np.histogram2d(moving_fake.numpy()[:,0], moving_fake.numpy()[:,1], (20,30), [x_range, y_range], normed=True)
            Hlarge = scipy.misc.imresize(H, 10*np.array(H.shape))
            last_itr_visuals.append(vis.image(np.flipud(Hlarge.T), opts={'title': 'Moving Fakes Heatmap'}))


            # fake_grid = vutils.make_grid(fixed_fake, padding=2, normalize=True)
            # moving_fake_grid = vutils.make_grid(moving_fake, padding=2, normalize=True)

            # last_itr_visuals.append(vis.image(fake_grid, opts={'title': 'Fixed Fakes [epoch][itr]: [%d/%d][%d/%d]' % (epoch, num_epochs, i, len(dataloader)) }))
            # last_itr_visuals.append(vis.image(moving_fake_grid, opts={'title': 'Moving Fakes [epoch][itr]: [%d/%d][%d/%d]' % (epoch, num_epochs, i, len(dataloader)) }))
            # save an image            
            # scipy.misc.imsave('%s/%06d.png' % (outdata_path, int(iters / 100)),
            #     np.moveaxis(fixed_fake[random.randint(0, fixed_fake.shape[0]-1),:,:,:].numpy(),0,-1)) 
            
            # plot some lines
            max_line_samples = 200
            ds = max(1,len(data1_D) // (max_line_samples+1))

            # losses
            # last_itr_visuals.append(vis.line(decimate(losses,ds), list(range(0,iters+1,ds)), opts={'legend': ['errG', 'errD'], 'title': 'Network Losses'}))
            # last_itr_visuals.append(vis.line(decimate(real_losses_detail,ds), list(range(0,iters+1,ds)), opts={'legend': ['d0g1', 'd2g1', 'd0g2', 'd1g2'], 'title': 'Real Data Losses'}))
            # last_itr_visuals.append(vis.line(decimate(fake_losses_detail,ds), list(range(0,iters+1,ds)), opts={'legend': ['d1g0', 'd2g0', 'd1g0_g', 'd2g0_g', 'd0g0_g'], 'title': 'Fake Data Losses'}))

            # network outputs
            output_legend = ['output_%d' % i for i in range(n_classes)]
            last_itr_visuals.append(vis.line(decimate(data1_D,ds), list(range(0,iters+1,ds)), opts={'legend': output_legend, 'title': 'Data1 classification'}))
            last_itr_visuals.append(vis.line(decimate(data2_D,ds), list(range(0,iters+1,ds)), opts={'legend': output_legend, 'title': 'Data2 classification'}))
            last_itr_visuals.append(vis.line(decimate(fake_D,ds), list(range(0,iters+1,ds)), opts={'legend': output_legend, 'title': 'Fake classification, D step'}))
            # last_itr_visuals.append(vis.line(decimate(fake_D_gen,ds), list(range(0,iters+1,ds)), opts={'legend': output_legend, 'title': 'Fake classification, G step'}))

            # itrs per second
            # if perf:
            #     ds = max(1,len(perf) // (max_line_samples+1))
            #     last_itr_visuals.append(vis.line(decimate(perf, ds), perftime[::ds], opts={'title': 'iters per second'}))
            
        iters += 1




######################################################################
# Where to Go Next
# ----------------
# 
# We have reached the end of our journey, but there are several places you
# could go from here. You could:
# 
# -  Train for longer to see how good the results get
# -  Modify this model to take a different dataset and possibly change the
#    size of the images and the model architecture
# -  Check out some other cool GAN projects
#    `here <https://github.com/nashory/gans-awesome-applications>`__
# -  Create GANs that generate
#    `music <https://deepmind.com/blog/wavenet-generative-model-raw-audio/>`__
# 

