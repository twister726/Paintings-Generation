#!/usr/bin/env python
# coding: utf-8

# In[1]:


from decoder import *
from encoder import *
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
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from matplotlib import pyplot as plt
import sys


# In[6]:




def validate_test(val_loader, encoder, decoder, criterion, maxSeqLen,
             vocab, batch_size, device):

    
    #Evaluation Mode
    decoder.eval()
    encoder.eval()
        
        
    with torch.no_grad():
        
        count    = 0
        loss_avg = 0
                
        for i, (inputs, labels, lengths) in enumerate(val_loader):
          
            
            # Move to device, if available
            if device is not None:
                inputs = inputs.to(device)
                labels = labels.to(device)

                        
            enc_out = encoder(inputs)

            decoder.resetHidden(inputs.shape[0])
            outputs = decoder(labels, enc_out, lengths)


            
            loss = criterion(outputs, labels.cuda(device))
            loss_avg += loss
            count+=1
            

            
        
        
            
            del labels
            del outputs            
  
        loss_avg  = loss_avg/count
        print('VAL: loss_avg: ', loss_avg)

        
        
        
            
    return loss_avg




def trainEncoderDecoder(encoder, decoder, criterion, epochs,
                        train_loader,val_loader, test_loader,
                        name, batch_size, maxSeqLen, vocab, device = None):
    
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
    pickle_file = logname_summary[::-4] +'.pkl'
    
    
    try:
        os.mkdir('./generated_imgs')
    except:
        pass
    
    generated_imgs_filename = './generated_imgs/generated_imgs' + name + '_summary' + str(i) + '.log'
    
    
    parameters = list(encoder.fc.parameters())
    parameters.extend(list(decoder.parameters()))
    optimizer = optim.Adam(parameters, lr=5e-5)
    

    if device is not None:    
        encoder.to(device)
        decoder.to(device)
        
        
    
    val_loss_set = []
    val_bleu1_set = []
    val_bleu4_set = []
    
    
    training_loss = []
    
    # Early Stop criteria
    minLoss = 1e6
    minLossIdx = 0
    earliestStopEpoch = 7
    earlyStopDelta = 3
    for epoch in range(epochs):
        ts = time.time()

                            
        for iter, (inputs, labels, lengths) in tqdm(enumerate(train_loader)):

            optimizer.zero_grad()
            
            
            encoder.train()
            decoder.train()
            
            if device is not None:
                inputs = inputs.to(device)# Move your inputs onto the gpu
                labels = labels.to(device) # Move your labels onto the gpu
            
                
            enc_out = encoder(inputs)
            
            temperature = 1
            
            
            
            decoder.resetHidden(inputs.shape[0])
            outputs = decoder(labels, enc_out, lengths) #calls forward

            loss = criterion(outputs, labels.cuda(device))
            del labels
            del outputs

            loss.backward()

            optimizer.step()

            if iter % 200 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss))
                
        print("epoch{}, iter{}, loss: {}, epoch duration: {}".format(epoch, iter, loss, time.time() - ts))
        test_pred = decoder.generate_caption(enc_out, maxSeqLen, temperature).cpu()
        
        k = 0
        for b in range(inputs.shape[0]):
            caption = (" ").join([vocab.idx2word[x.item()] for x in test_pred[b]])
            img = tf.ToPILImage()(inputs[b,:,:,:].cpu())
            plt.imshow(img)
                    
            plt.show()
            print("Caption: " + caption)
            
        
        # calculate val loss each epoch
        val_loss  = validate_test(val_loader, encoder, decoder, criterion,maxSeqLen,
                             vocab, batch_size, device).item()
        val_loss_set.append(val_loss)

  
        training_loss.append(loss)
        
        torch.save(encoder, 'weights/' + name + 'encoder_epoch{}'.format(epoch))
        torch.save(decoder, 'weights/'+ name + 'decoder_epoch{}'.format(epoch))
        
        with open(logname, "a") as file:
            file.write("writing!\n")
            file.write("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
            file.write("\n training Loss:   " + str(loss.item()))
            file.write("\n Validation Loss: " + str(val_loss_set[-1]))
                                         
                                                                                                
                                                                                                
        
        # Early stopping
        if val_loss < minLoss:
            # Store new best
            minLoss = val_loss#.item()
            minLossIdx = epoch
            torch.save(encoder, 'weights/' + name + 'encoder_best')
            torch.save(decoder, 'weights/'+ name + 'decoder_best')
            
        #If passed min threshold, and no new min has been reached for delta epochs
        elif epoch > earliestStopEpoch and (epoch - minLossIdx) > earlyStopDelta:
            print("Stopping early at {}".format(minLossIdx))
            
            break
        

        
        
        with open(logname_summary, "w") as file:
            file.write("Summary!\n")
            file.write("\n training Loss:   " + str(training_loss))        
            file.write("\n Validation Loss : " + str(val_loss_set))

    #return val_loss_set, val_acc_set, val_iou_set


# In[8]:


if __name__=='__main__':
    name = "lstm128"
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        if len(sys.argv) > 2:
            device = torch.device("cuda:" + sys.argv[2])

    random.seed(24)        
    if len(sys.argv) > 1:
        name = sys.argv[1]

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
    
    # Will shuffle the trainIds incase of ordering in csv
    random.shuffle(trainIds)
    splitIdx = int(len(trainIds)/5)
    
    # Selecting 1/5 of training set as validation
    valIds = trainIds[:splitIdx]
    trainIds = trainIds[splitIdx:]
    #print(trainIds)
    
    
    trainValRoot = "./data/realImages/"
    testRoot = "./data/realImages/"
    
    trainValJson = "./data/captions/trainvalCaps.csv"
    testJson = "./data/captions/testCaps.csv"
    
    
    with open('./data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    img_side_length = 128
    transform = tf.Compose([
        tf.Resize(img_side_length),
        #tf.RandomCrop(img_side_length),
        tf.CenterCrop(img_side_length),
        tf.ToTensor(),
    ])
    batch_size = 20
    shuffle = True
    num_workers = 20
    
    
    trainDl = get_loader(trainValRoot, trainValJson, trainIds, vocab, 
                         transform=transform, batch_size=batch_size, 
                         shuffle=shuffle, num_workers=num_workers)
    valDl = get_loader(trainValRoot, trainValJson, valIds, vocab, 
                         transform=transform, batch_size=batch_size, 
                         shuffle=shuffle, num_workers=num_workers)
    testDl = get_loader(testRoot, testJson, testIds, vocab, 
                        transform=transform, batch_size=batch_size, 
                        shuffle=shuffle, num_workers=num_workers)
    
    encoded_feature_dim = 800
    maxSeqLen = 56
    hidden_dim = 800
    depth = 1
    
    encoder = Encoder(encoded_feature_dim)
    # Turn off all gradients in encoder
    for param in encoder.parameters():
        param.requires_grad = False
    # Turn on gradient of final hidden layer for fine tuning
    for param in encoder.fc.parameters():
        param.requires_grad = True
    
    
    if name=="rnn":
        decoder = RNNDecoder(encoded_feature_dim, hidden_dim, depth, vocab.idx, batch_size, device)
    else:
        decoder = Decoder(encoded_feature_dim, hidden_dim, depth, vocab.idx, batch_size)
    

    criterion = nn.CrossEntropyLoss()
    
    epochs = 100
    trainEncoderDecoder(encoder, decoder, criterion, epochs,
                        trainDl, valDl, testDl, name,
                        batch_size, maxSeqLen, vocab, device)









