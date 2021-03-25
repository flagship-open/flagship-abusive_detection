#-*- encoding: utf-8 -*-
import unicodedata
import string
import re
import random
import time
import math
from collections import defaultdict
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.metrics import roc_auc_score
USE_CUDA = False


num_layer=1


embedding_dim = 300
num_filters = 100
kernel_sizes = [3,4,5]


class AttentionWordRNN(nn.Module):
    """
    The embedding layer + CNN model that will be used to perform sentiment analysis.
    """

    def __init__(self,  
                 num_filters=128, kernel_sizes=[3, 4, 5], freeze_embeddings=True, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(AttentionWordRNN, self).__init__()

        # set class vars
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim
        
        # 1. embedding layer
        
        # 2. convolutional layers
        KK=[]
        for K in kernel_sizes:
            KK.append( K + 1 if K % 2 == 0 else K)
            

        self.convs_1d = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim), padding=(k//2,0)) 
            for k in KK])

        # 3. final, fully-connected layer for classification
        self.lstm = nn.LSTM(num_filters,
                           num_filters,num_layer,batch_first = True,bidirectional= True)
        
        # 4. dropout and sigmoid layers
        self.dropout = nn.Dropout(drop_prob)
        self.sig = nn.Sigmoid()
    
    def conv_and_pool(self, x, conv,x1):

        x = conv(x)
        x = F.relu(x)
        x = x.squeeze(3)
        return x

    def forward(self, x):
        x = torch.transpose(x,1,0)
        embeds = x.unsqueeze(1)
        conv_results = [self.conv_and_pool(embeds, conv,x) for conv in self.convs_1d]
        x = torch.cat(conv_results, 0)
        x = torch.transpose(x, 1,2)
        output,(final_hidden_state, final_cell_state) =self.lstm(x)
        x = torch.transpose(output, 0,1)
        x = torch.transpose(x, 1,2)
        x = F.max_pool1d(x, x.size(2))
        x = x.squeeze(2)
        return x.unsqueeze(0)
      




class AttentionSentRNN(nn.Module):
    def __init__(self, 
                 num_filters=128, kernel_sizes=[3, 4, 5], freeze_embeddings=True, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(AttentionSentRNN, self).__init__()

        self.num_filters = num_filters
        #self.embedding_dim = embedding_dim


        KK=[]
        for K in kernel_sizes:
            KK.append( K + 1 if K % 2 == 0 else K)
            
        self.convs_1d = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, 2*num_filters), padding=(k//2,0)) 
            for k in KK])
        

        self.fc1= nn.Linear(384, 1000) 
        self.lstm = nn.LSTM(num_filters, num_filters,num_layer,batch_first = True,bidirectional= True)
        self.dropout = nn.Dropout(drop_prob)
        self.sig = nn.Sigmoid()
    
    def conv_and_pool(self, x, conv,x1):
        x = conv(x)
        x = F.relu(x)
        x = x.squeeze(3)
        return x

    def forward(self, x):
        embeds = x.unsqueeze(1)
        conv_results = [self.conv_and_pool(embeds, conv,x) for conv in self.convs_1d]
        x = torch.cat(conv_results, 0)
        x = torch.transpose(x, 1,2)
        output,(final_hidden_state, final_cell_state) =self.lstm(x)
        x = torch.transpose(output, 0,1)
        x = torch.transpose(x, 1,2)
        x = F.max_pool1d(x, x.size(2))

        return x.squeeze(2)
      
                
      

