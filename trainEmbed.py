#!/usr/bin/env python
# coding: utf-8

# In[1]:


from textEncoder import *
from other_data_loader import *
import pickle
import random
import torch.optim as optim
from torch.autograd import Variable
import csv
import time
from tqdm import tqdm
import gc
import os
import torchvision.transforms as tf
import json
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from matplotlib import pyplot as plt


# In[2]:




def validate_test(val_loader, embed, criterion, maxSeqLen,
             vocab, batch_size, use_gpu = True, calculate_bleu = True):

    
    #Evaluation Mode
    embed.eval()

    
    references = list()
    hypotheses = list() 
   
    if use_gpu:
        device = torch.device("cuda:0")
        
        
    with torch.no_grad():
        
        count    = 0
        loss_avg = 0

                
        for i, (inputs, caps, actual_lengths) in enumerate(val_loader):
            
            
            
            # Move to device, if available
            if use_gpu:
                inputs = inputs.to(device)
                caps = caps.to(device)


            
            embed.resetHidden(inputs.shape[0])
            outputs, h, s = embed(caps, actual_lengths)
            
            
            loss = criterion(outputs, caps[:,1:])
            loss_avg += loss.item()
            count+=1
            
            
            
            del caps
            del outputs            
            

                
        loss_avg  = loss_avg/count
        print('VAL: loss_avg: ', loss_avg)     
        
            
    return loss_avg


# In[3]:


def trainEmbed(embed, criterion, epochs, train_loader,val_loader, test_loader, 
               name, batch_size, maxSeqLen, vocab,save_generated_imgs= False):
    
    #Create non-existing logfiles
    logname = './logs/' + name + '.log'
    i = 0
    if os.path.exists(logname) == True:
        
        logname = './logs/' + name + str(i) + '.log'
        while os.path.exists(logname):
            i+=1
            logname = './logs/' + name + str(i) + '.log'

    print('Loading results to logfile: ' + logname)
    with open(logname, "w") as file:
        file.write("Log file DATA: Validation Loss and Accuracy\n") 
    
    logname_summary = './logs/' + name + '_summary' + str(i) + '.log'    
    print('Loading Summary to : ' + logname_summary) 
    
    
    try:
        os.mkdir('./generated_imgs')
    except:
        pass
    
    generated_imgs_filename = './generated_imgs/generated_imgs' + name + '_summary' + str(i) + '.log'
    
    
    
    parameters = list(embed.parameters())
    optimizer = optim.Adam(parameters, lr=5e-5)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda:0")
#         encoder = torch.nn.DataParallel(encoder)
#         decoder = torch.nn.DataParallel(decoder)
        
        embed.to(device)
        
        
    
    val_loss_set = []

    
    
    training_loss = []
    
    # Early Stop criteria
    minLoss = 1e6
    minLossIdx = 0
    earliestStopEpoch = 7
    earlyStopDelta = 3
    for epoch in range(epochs):
        ts = time.time()

        for iter, (inputs, labels, lengths) in tqdm(enumerate(train_loader)):
            del inputs
            optimizer.zero_grad()
            
            
            
            embed.train()
            
            
            if use_gpu:
                labels = labels.to(device) # Move your labels onto the gpu
            
                

            embed.resetHidden(labels.shape[0])
            outputs, w, s = embed(labels, lengths) #calls forward

            loss = criterion(outputs, labels[:,1:])
            del labels
            del outputs

            loss.backward()
#             loss = loss#.item()
            optimizer.step()
            loss = loss.item()
            if iter % 200 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss))

                
        print("epoch{}, iter{}, loss: {}, epoch duration: {}".format(epoch, iter, loss, time.time() - ts))
        torch.save(embed, 'weights/' + name + 'embed_epoch{}'.format(epoch))



        
        # calculate val loss each epoch
        val_loss = validate_test(val_loader, embed, criterion,maxSeqLen,
                             vocab, batch_size, use_gpu, calculate_bleu = False)
        val_loss_set.append(val_loss)


     
        training_loss.append(loss)
        
        # Early stopping
        if val_loss < minLoss:
            # Store new best
            minLoss = val_loss#.item()
            minLossIdx = epoch
            torch.save(embed, 'weights/' + name + '_embed_best')

            
        #If passed min threshold, and no new min has been reached for delta epochs
        elif epoch > earliestStopEpoch and (epoch - minLossIdx) > earlyStopDelta:
            print("Stopping early at {}".format(minLossIdx))
            
            break
        

        
        
        with open(logname_summary, "w") as file:
            file.write("Summary!\n")
            file.write("\n training Loss:   " + str(training_loss))        
            file.write("\n Validation Loss : " + str(val_loss_set))
            


# In[ ]:


if __name__=='__main__':
    with open('trainvalIds.csv', 'r') as f:
        trainIds = []
        for line in f:
            if len(line) > 1:
                trainIds.append(line.strip("\n"))

        
    with open('testIds.csv', 'r') as f:
        testIds = []
        for line in f:
            if len(line) > 1:
                testIds.append(line.strip("\n"))
    
    print("found {} train ids".format(len(trainIds)))
    print("found {} test ids".format(len(testIds)))
    
    # Will shuffle the trainIds incase of ordering in csv
    random.shuffle(trainIds)
    splitIdx = int(len(trainIds)/5)
    
    # Selecting 1/5 of training set as validation
    valIds = trainIds[:splitIdx]
    trainIds = trainIds[splitIdx:]
    #print(trainIds)
    
    
    trainValRoot = "./data/realImages/"
    testRoot = "./data/realImages/"
    
    trainValCaps = "./data/captions/trainvalCaps.csv"
    testCaps = "./data/captions/testCaps.csv"
    
    
    with open('./data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    img_side_length = 256
    transform = tf.Compose([
        tf.Resize(img_side_length),
        #tf.RandomCrop(img_side_length),
        tf.CenterCrop(img_side_length),
        tf.ToTensor(),
    ])
    batch_size = 30
    shuffle = True
    num_workers = 30
    
    
    trainDl = get_loader(trainValRoot, trainValCaps, trainIds, vocab, 
                         transform=transform, batch_size=batch_size, 
                         shuffle=shuffle, num_workers=num_workers)
    valDl = get_loader(trainValRoot, trainValCaps, valIds, vocab, 
                         transform=transform, batch_size=batch_size, 
                         shuffle=shuffle, num_workers=num_workers)
    testDl = get_loader(testRoot, testCaps, testIds, vocab, 
                        transform=transform, batch_size=batch_size, 
                        shuffle=shuffle, num_workers=num_workers)
    
    encoded_feature_dim = 800
    maxSeqLen = 49
    hidden_dim = 256

    
    
    embed = Embed(encoded_feature_dim, hidden_dim, vocab, batch_size)
    
#     criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    
    epochs = 100
    trainEmbed(embed, criterion, epochs,
                        trainDl, valDl, testDl, "base",
                        batch_size, maxSeqLen, vocab,save_generated_imgs = True)


# In[ ]:


get_ipython().run_line_magic('debug', '')


# In[ ]:


with open('./data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
vocab.idx


# In[ ]:


x = torch.Tensor([])
