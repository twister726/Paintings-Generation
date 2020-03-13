import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO
from torchvision import transforms


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, ids, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = self.coco.getAnnIds(ids)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        
       
        
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        maxCaptionSize = 56
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        pad = '<pad>'
        
        tokens = tokens[:(maxCaptionSize)]
        tokens.insert(0, '<start>')
        tokens.append('<end>')
        length = len(tokens)
        if len(tokens) < maxCaptionSize:
            tokens.extend([pad]*(maxCaptionSize - len(tokens)))
        
        caption = [vocab(token) for token in tokens[:maxCaptionSize]]

        target = torch.Tensor(caption)

        return image, target, length

    def __len__(self):
        return len(self.ids)
    
class FlowerDataset(data.Dataset):
    def __init__(self, root, caps, ids, vocab, transform=None):
        self.root = root
        self.img2cap = {}
        self.cap2cap = {}
        
        self.maxCapLen = -1
#         print("ids: ")
#         print(ids[:10])
        with open(caps, 'r') as f:
            for l in f:
                split = l.split(',')
                if len(split) < 3:
#                     print("Weirdooo" + l)
                    continue
                img_id = split[0]
                cap_id = split[1]
                cap = split[2]
                
                if img_id in ids:
                    self.cap2cap[cap_id] = cap
                    if len(cap.split(' ')) > self.maxCapLen:
                        self.maxCapLen = len(cap.split(' '))
                if img_id in self.img2cap:
                    self.img2cap[img_id].append(cap_id)
                else:
                    self.img2cap[img_id] = [cap_id]
                    
        self.ids = list(self.cap2cap.keys())
        print("# ids: {}".format(len(self.ids)))
                
                
        
        self.vocab = vocab
        self.transform = transform
        self.maxCapLen += 2

        
    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        vocab = self.vocab
        cap_id = self.ids[index]
        
       
        
        caption = self.cap2cap[cap_id]
        img_id = cap_id[:-1]
        
        path = "image_" + img_id + ".jpg"

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        #maxCaptionSize = 56
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        pad = '<pad>'
        
        tokens = tokens[:(self.maxCapLen)]
        tokens.insert(0, '<start>')
        tokens.append('<end>')
        length = len(tokens)
        if length < self.maxCapLen:
            tokens.extend([pad]*(self.maxCapLen - length))
        
        caption = [vocab(token) for token in tokens[:self.maxCapLen]]

        target = torch.Tensor(caption)

        return image, target, length

    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    #print(data)
#     print("Data Loader data")
#     print("Samples: {}".format(len(data)))
#     print("data each: {}".format(len(data[0])))
#     print("data 2 ")
#     print(data[0][2])
#     for d in data[0]:
#         print(d.shape)
    data.sort(key=lambda x: x[2], reverse=True)
    images, captions, lengths = zip(*data)
    
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    targets = torch.zeros(len(captions), len(data[0][1])).long()
    for i, cap in enumerate(captions):
        targets[i, :lengths[i]] = cap[:lengths[i]]        
    return images, targets, lengths


                

def get_loader(root, caps, ids, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
#     coco = CocoDataset(root=root,
#                        json=json,
#                        ids = ids,
#                        vocab=vocab,
#                        transform=transform)
    flower = FlowerDataset(root=root,
                       caps = caps,
                       ids = ids,
                       vocab=vocab,
                       transform=transform)
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=flower, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
