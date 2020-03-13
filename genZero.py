import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F

class BaseGenerator(nn.Module):
    def __init__(self, batch_size, embedding_dim):
        super(BaseGenerator, self).__init__()
        
        self.batch_size = batch_size
        #self.hidden_dim = hidden_dim
        self.device = torch.device("cuda:0")
        self.embedding_dim = embedding_dim
#        self.initial_fc = nn.Linear(encoded_feature_dim, embedding_dim)
        
        # Embed input vector into tensor
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)

        #self.hidden_dim = hidden_dim
        # output 3x8x8 -> 3x16x16 -> 3x32x32 -> 3x64x64
        self.F0_FC = nn.Linear(self.embedding_dim, 8*8*3)
        # Generates h0
        self.F0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels = 3, out_channels= 3, kernel_size = 3, stride = 1,padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels = 3, out_channels= 3, kernel_size = 3, stride = 1,padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(inplace=True),
        )
        
        self.G0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels = 3, out_channels= 3, kernel_size = 3, stride = 1,padding=1),
        )
        
        self.D0_compress = nn.Linear(self.embedding_dim, 128)
        
        self.D0_down = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels= 64, kernel_size = 4, stride = 2,padding=1),
            #No batch norm/leaky Relu on first
            
            nn.Conv2d(in_channels = 64, out_channels= 128, kernel_size = 4, stride = 2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(in_channels = 128, out_channels= 256, kernel_size = 4, stride = 2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(in_channels = 256, out_channels= 512, kernel_size = 4, stride = 2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )
        
        self.D0_join = nn.Conv2d(in_channels = 640, out_channels = 50, kernel_size=1,stride=1)
        self.D0_judge = nn.Sequential(
            nn.Linear(800, 1),
            #nn.Softmax(),
        )

    def setDiscriminatorGrad(self, grad_on):
        discParameters = list(self.D0_down.parameters())
        discParameters.extend(self.D0_compress.parameters())
        discParameters.extend(self.D0_join.parameters())
        discParameters.extend(self.D0_judge.parameters())
        for param in discParameters:
            param.requires_grad = grad_on
                   
    def forward(self, sentence, imgs):


        resized = self.F0_FC(sentence)
        h0 = self.F0(resized.view(imgs.shape[0], 3, 8, 8))
        gen_img = self.G0(h0)
        d_down = self.D0_down(gen_img)
        d_comp = self.D0_compress(sentence)
        
        #Reshape to concat and join
        stack = torch.stack([d_comp.clone().squeeze(0), d_comp.clone().squeeze(0),
                             d_comp.clone().squeeze(0), d_comp.clone().squeeze(0)], dim=2)
        stacked = torch.stack([stack.clone(), stack.clone(), stack.clone(), stack.clone()], dim=3)
        
        d_down_real = self.D0_down(imgs)
        stackedReal = stacked.clone()
        
        
        joined_gen = self.D0_join(torch.cat([stacked, d_down], dim = 1)).view(imgs.shape[0], 800)
        joined_real = self.D0_join(torch.cat([stackedReal, d_down_real], dim = 1)).view(imgs.shape[0], 800)
        
        return gen_img, h0, self.D0_judge(joined_gen), self.D0_judge(joined_real)
    

