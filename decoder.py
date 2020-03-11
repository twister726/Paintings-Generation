import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, depth, vocab_size, batch_size):
        super(Decoder, self).__init__()
        self.depth = depth
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda:0")
        
#        self.initial_fc = nn.Linear(encoded_feature_dim, embedding_dim)
        
        # Embed input vector into tensor
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.hidden_dim = hidden_dim
        # LSTM layer will take the feature tensor and previous hidden layer as input

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, depth, batch_first=True)
        self.hidden = (torch.zeros((depth, batch_size, hidden_dim)),
                        torch.zeros((depth, batch_size, hidden_dim)))

        # Linear layer mapping hidden rep -> output vector
        # We will use the full vocab in output as well
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def resetHidden(self, batch_size):
        self.hidden = (torch.zeros((self.depth, batch_size, self.hidden_dim)).to(self.device),
                        torch.zeros((self.depth, batch_size, self.hidden_dim)).to(self.device))
                       
    def forward(self, sentence, features, lengths):

        embeds = self.embedding(sentence)
        
        #lstm_inp = torch.cat((features.unsqueeze(1), embeds), 1)

        #packed_sequence = pack_padded_sequence(lstm_inp, lengths, batch_first=True)
        #packed_sequence = pack_padded_sequence(lstm_inp, lengths, batch_first=True, enforce_sorted=False)
        
        
        #packed_out, self.hidden = self.lstm(packed_sequence, self.hidden)
        outputs = []
        lstmOut, self.hidden = self.lstm(features.unsqueeze(1), self.hidden)
        outputs.append(lstmOut)
        for i in range(embeds.shape[1] - 1):
            lstmOut, self.hidden = self.lstm(embeds[:,i,:].unsqueeze(1), self.hidden)
            outputs.append(lstmOut)
#        print("outputs: {}".format(outputs[0].shape))
#        print("Ouputs stacked: {}".format(torch.stack(outputs, 1).shape))
        fc_out = self.fc(torch.stack(outputs, 1).squeeze(2))
        
        #print("FC out; {}".format(fc_out.shape))
        _, indices = fc_out.max(2)
        #word_scores = fc_out
        #return indices.type(torch.int64).cuda()
        return fc_out.permute([0,2,1])
    
    def generate_caption(self, features, maxSeqLen, temperature, stochastic=False):
        # TODO - function for generating caption without using teacher forcing (using network outputs)
        
#        features = self.initial_fc(features)
#         print('features shape: ', features.size())
#         print('maxseqlen: ', maxSeqLen)
        lstm_inp = features.unsqueeze(1)
        word_ids = []
        self.resetHidden(1)
        if stochastic:
            for i in range(maxSeqLen):
                lstm_out, _ = self.lstm(lstm_inp)
                fc_out = self.fc(lstm_out.squeeze(1))
                scores = F.softmax(fc_out, dim=1) / temperature
                indices = (torch.distributions.Categorical(scores)).sample()
                word_ids.append(indices)
                lstm_inp = self.embedding(indices).unsqueeze(1)
                
                
        else:
            for i in range(maxSeqLen):
                if i == 0:
                    lstm_out, hidden = self.lstm(lstm_inp)
                else:
#                     print("Hidden shape")
#                     print(hidden[0].shape)
#                     print(hidden[1].shape)
                    lstm_out, hidden = self.lstm(lstm_inp, hidden)
                    
#                 print("Output size")
#                 print(lstm_out.data.shape)
                fc_out = self.fc(lstm_out.squeeze(1))
                _, indices = fc_out.max(1)

                
                lstm_inp = self.embedding(indices).unsqueeze(1)
#                 print("Input shape: ")
#                 print(lstm_inp.shape)
                word_ids.append(indices.cpu())
                
            
        word_ids = torch.stack(word_ids,1)
#         print('word ids shape: ', word_ids.size())
        #        print('word ids shape: ', word_ids.size())
#        print('word ids: ', word_ids)
        
        return word_ids
