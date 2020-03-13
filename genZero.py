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
        self.embedding_dim = 256
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
        
        self.D0_down = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels= 64, kernel_size = 4, stride = 2,padding=2),
            #No batch norm/leaky Relu on first
            
            nn.Conv2d(in_channels = 64, out_channels= 128, kernel_size = 4, stride = 2,padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(in_channels = 128, out_channels= 256, kernel_size = 4, stride = 2,padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(in_channels = 256, out_channels= 512, kernel_size = 4, stride = 2,padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )
        
        self.D0_join = nn.Conv2d(in_channels = 768, out_channels = 50, kernel_size=1,stride=1)
        self.D0_judge = Linear(800, 1)


                   
    def forward(self, sentence):

#         embeds = self.embedding(sentence)
        
        #lstm_inp = torch.cat((features.unsqueeze(1), embeds), 1)

        #packed_sequence = pack_padded_sequence(lstm_inp, lengths, batch_first=True)
        #packed_sequence = pack_padded_sequence(lstm_inp, lengths, batch_first=True, enforce_sorted=False)
        
        
        #packed_out, self.hidden = self.lstm(packed_sequence, self.hidden)
#         outputs = []
#         lstmOut, self.hidden = self.lstm(features.unsqueeze(1), self.hidden)
#         outputs.append(lstmOut)
#         for i in range(embeds.shape[1] - 1):
#             lstmOut, self.hidden = self.lstm(embeds[:,i,:].unsqueeze(1), self.hidden)
#             outputs.append(lstmOut)
#        print("outputs: {}".format(outputs[0].shape))
#        print("Ouputs stacked: {}".format(torch.stack(outputs, 1).shape))
#         fc_out = self.fc(torch.stack(outputs, 1).squeeze(2))
        
        #print("FC out; {}".format(fc_out.shape))
#         _, indices = fc_out.max(2)
        #word_scores = fc_out
        #return indices.type(torch.int64).cuda()
        resized = self.F0_FC(sentence)
        h0 = self.F0(resized)
        gen_img = self.G0(h0)
        d_down = self.D0_down(gen_img)
        
        #Reshape to concat and join
        stack = torch.stack([sentence.clone(), sentence.clone(), sentence.clone(), sentence.clone()], dim=2)
        stacked = torch.stack([stack.clone(), stack.clone(), stack.clone(), stack.clone()], dim=3)
        
        joined = self.G0_join(torch.stack([stacked, d_down], dim = 1))
        return gen_img, h0, self.D0_judge(joined)
    
#     def generate_caption(self, features, maxSeqLen, temperature, stochastic=False):
#         # TODO - function for generating caption without using teacher forcing (using network outputs)
        
# #        features = self.initial_fc(features)
# #         print('features shape: ', features.size())
# #         print('maxseqlen: ', maxSeqLen)
#         lstm_inp = features.unsqueeze(1)
#         word_ids = []
#         self.resetHidden(1)
#         if stochastic:
#             for i in range(maxSeqLen):
#                 lstm_out, _ = self.lstm(lstm_inp)
#                 fc_out = self.fc(lstm_out.squeeze(1))
#                 scores = F.softmax(fc_out, dim=1) / temperature
#                 indices = (torch.distributions.Categorical(scores)).sample()
#                 word_ids.append(indices)
#                 lstm_inp = self.embedding(indices).unsqueeze(1)
                
                
#         else:
#             for i in range(maxSeqLen):
#                 if i == 0:
#                     lstm_out, hidden = self.lstm(lstm_inp)
#                 else:
# #                     print("Hidden shape")
# #                     print(hidden[0].shape)
# #                     print(hidden[1].shape)
#                     lstm_out, hidden = self.lstm(lstm_inp, hidden)
                    
# #                 print("Output size")
# #                 print(lstm_out.data.shape)
#                 fc_out = self.fc(lstm_out.squeeze(1))
#                 _, indices = fc_out.max(1)

                
#                 lstm_inp = self.embedding(indices).unsqueeze(1)
# #                 print("Input shape: ")
# #                 print(lstm_inp.shape)
#                 word_ids.append(indices.cpu())
                
            
#         word_ids = torch.stack(word_ids,1)
# #         print('word ids shape: ', word_ids.size())
#         #        print('word ids shape: ', word_ids.size())
# #        print('word ids: ', word_ids)
        
#        return word_ids
