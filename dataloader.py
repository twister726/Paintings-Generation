from torch.utils.data import Dataset, DataLoader# For custom data-sets
import torchvision.transforms as transforms
import numpy as np
import os
import math
from PIL import Image
import torch

class FlowerDataset(Dataset):
    
    def __init__(self, img_dir='./data/102flowers', caption_dir='./data/final_captions', data_type='train'):
        self.all_imgs = []
        self.all_captions = []
        self.vector_captions = []
        
        self.vocab = set()
        
        train_frac = 0.75
        val_frac = 0.2
        
        captions_per_img = 5
        
        imnames = os.listdir(img_dir)
        
        train_ind = int(math.floor(train_frac*len(imnames)))
        val_ind = int(math.floor((train_frac+val_frac)*len(imnames)))
        
        if data_type == 'train':
            desired_imnames = imnames[:train_ind]
        elif data_type == 'val':
            desired_imnames = imnames[train_ind:val_ind]
        else:
            desired_imnames = imnames[val_ind:]
        
        for imname in desired_imnames[:10]:
            impath = os.path.join(img_dir, imname)
            im = np.asarray(Image.open(impath))
            im = np.swapaxes(np.swapaxes(im,0,2), 1, 2)
#             self.all_imgs.append(im)
            
            capname = imname.split('.')[0]+'.txt'
            cappath = os.path.join(caption_dir, capname)
            img_captions = []
            with open(cappath, 'r') as f:
                img_captions = [x.rstrip('\n') for x in f.readlines()]
                
            for caption in img_captions[:captions_per_img]:
                self.all_imgs.append(im)
                self.all_captions.append(caption)
                for word in caption.split():
                    self.vocab.add(word)
        
        self.vocab = list(self.vocab)
        
        for caption in self.all_captions:
            self.vector_captions.append(np.array([self.vocab.index(word) for word in caption.split()]))
            
        print(self.vector_captions[:10])
        print('hi ', self.__getitem__(2))
        
    def __getitem__(self, idx):
        return torch.Tensor(self.all_imgs[idx]), torch.Tensor(self.vector_captions[idx])
            
# f = FlowerDataset()