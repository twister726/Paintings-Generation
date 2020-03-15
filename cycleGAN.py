import time
import datetime
import sys

import numpy as np

import glob
import random
import os

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import argparse
import itertools

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

#what even: track training
class Logger():
    def __init__(self, n_epochs, batches_epoch):
        #self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()   #data[0]
            else:
                self.losses[loss_name] += losses[loss_name].item()    #data[0]

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))
        
        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            
            # Plot losses
            for loss_name, loss in self.losses.items():
                #if loss_name not in self.loss_windows:
                 #   self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
                 #                                                   opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                #else:
                    #self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0 # ?
            
            
            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

        
#discriminator is trained on buffer images rather than immediate output of generator
class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), "black hole"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

#lr update
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "decay should start earlier"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
        

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*')) #datasets titled A and B
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

def myimshow(image, ax=plt):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    return h

size = 256 #crop size
dataroot = './vangogh2photo/'
transforms_ = [ transforms.Resize(int(size*1.12), Image.BICUBIC), 
                transforms.CenterCrop(size), 
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]


train_test = ImageDataset(dataroot, transforms_=transforms_, unaligned=True)
x = train_test.__getitem__(index = 200)

#used within the generator
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        #initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        #downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        #residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        #upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        #output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        #series of conv layers
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        #classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        #average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    

epoch = 0
n_epochs = 100
batchSize = 1
dataroot = './vangogh2photo/'
lr = 0.001 
decay_epoch = 1#decay in lr starts after this epoch
size = 256 #crop size
input_nc = 3 #number of input channels
output_nc = 3 #number of output channels
n_cpu = 0 #number of cpu threads to use during batch generation
last_epoch = 0 #checkpoint usage

#training

#models
netG_A2B = Generator(input_nc, output_nc)
netG_B2A = Generator(output_nc, input_nc)
netD_A = Discriminator(input_nc)
netD_B = Discriminator(output_nc)

netG_A2B.to(device)
netG_B2A.to(device)
netD_A.to(device)
netD_B.to(device)

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

#losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

#optimizers & lr schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=4 * lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=4 * lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

#input & target memory allocation
Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.Tensor
input_A = Tensor(batchSize, input_nc, size, size)
input_B = Tensor(batchSize, output_nc, size, size)
target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

#dataset loader
transforms_ = [ transforms.Resize(int(size*1.12), Image.BICUBIC), 
                transforms.CenterCrop(size), 
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

dataloader = DataLoader(ImageDataset(dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=batchSize, shuffle=True, num_workers=n_cpu)

#loss track and #plot
logger = Logger(n_epochs, len(dataloader))


netG_A2B.load_state_dict(torch.load('./netG_A2B/checkpoint.pth.tar')['state_dict'])
netG_B2A.load_state_dict(torch.load('./netG_B2A/checkpoint.pth.tar')['state_dict'])
netD_A.load_state_dict(torch.load('./netD_A/checkpoint.pth.tar')['state_dict'])
netD_B.load_state_dict(torch.load('./netD_B/checkpoint.pth.tar')['state_dict'])
   
optimizer_G.load_state_dict(torch.load('./netG_A2B/checkpoint.pth.tar')['optimizer'])
optimizer_D_A.load_state_dict(torch.load('./netD_A/checkpoint.pth.tar')['optimizer'])
optimizer_D_B.load_state_dict(torch.load('./netD_B/checkpoint.pth.tar')['optimizer'])
epoch = torch.load('./netD_A/checkpoint.pth.tar')['epoch']


loss_track_g = []
loss_track_gidentity = []
loss_track_gcycle = []
loss_track_gan = []
loss_track_d = []
for epoch in range(epoch, n_epochs):
    #print(epoch)
    
    for i, batch in enumerate(dataloader):
        #set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        
        ##################################
        #generators A2B and B2A
        optimizer_G.zero_grad()

        #identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0 
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        #GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        #cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        #total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()
        ####################################

        #discriminator A: between real and fake (generated) A
        optimizer_D_A.zero_grad()

        #real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        #fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        #total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        #print(loss_D_A)
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        #discriminator B
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        #print(loss_D_B)
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        #logger report
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}, 
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})
        loss_track_g.append(loss_G)
        loss_track_gidentity.append(loss_identity_A + loss_identity_B)
        loss_track_gcycle.append(loss_cycle_ABA + loss_cycle_BAB)
        loss_track_gan.append(loss_GAN_A2B + loss_GAN_B2A)
        loss_track_d.append(loss_D_A + loss_D_B)

    #update lr
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    #save model checkpoints
    os.makedirs('./netG_A2B', exist_ok=True)
    checkpoint_pathGAB = os.path.join('./netG_A2B', "checkpoint.pth.tar")
    torch.save({'epoch': epoch + 1, 'state_dict': netG_A2B.state_dict(), 'optimizer': optimizer_G.state_dict()}, checkpoint_pathGAB)
    os.makedirs('./netG_B2A', exist_ok=True)
    checkpoint_pathGBA = os.path.join('./netG_B2A', "checkpoint.pth.tar")
    torch.save({'epoch': epoch + 1, 'state_dict': netG_B2A.state_dict(), 'optimizer': optimizer_G.state_dict()}, checkpoint_pathGBA)
    os.makedirs('./netD_A', exist_ok=True)
    checkpoint_pathDA = os.path.join('./netD_A', "checkpoint.pth.tar")
    torch.save({'epoch': epoch + 1, 'state_dict': netD_A.state_dict(), 'optimizer': optimizer_D_A.state_dict()}, checkpoint_pathDA)
    os.makedirs('./netD_B', exist_ok=True)
    checkpoint_pathDB = os.path.join('./netD_B', "checkpoint.pth.tar")
    torch.save({'epoch': epoch + 1, 'state_dict': netD_B.state_dict(), 'optimizer': optimizer_D_B.state_dict()}, checkpoint_pathDB)
    with open('losses.txt', 'a') as file:
            #file.write("writing!\n")
            file.write("Finish epoch {}".format(epoch + 1))
            file.write('loss_G   ' + str(loss_G.item()))
            file.write('loss_G_I   ' + str((loss_identity_A + loss_identity_B).item()))
            file.write('loss_G_cyc   ' + str((loss_cycle_ABA + loss_cycle_BAB).item()))
            file.write('loss_GAN   ' + str((loss_GAN_A2B + loss_GAN_B2A).item()))
            file.write('loss_D   ' + str((loss_D_A + loss_D_B).item()))
            
