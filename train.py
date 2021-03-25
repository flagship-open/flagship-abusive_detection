#-*- coding: utf-8 -*-

import json, re, abusive, socket, context_abusive
from flask import Flask, render_template, request
import flask

#ipaddress = socket.gethostbyname(socket.gethostname())
ipaddress='0.0.0.0'
app = Flask(__name__)

adict_name = 'data/abusive_dictionary.txt'
non_adict_name = 'data/abusive_dictionary2.txt'

abusive_dict_set = abusive.set_dict(adict_name)
whitelist_dict_set = abusive.set_dict(non_adict_name)

import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import torch._utils
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
import csv
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


embedding_dim = 300
num_filters = 100
kernel_sizes = [3,4,5]
num_layer=1




def get_predictions(val_tokens1, val_tokens2,  sent_attn):
    max_sents, batch_size, max_tokens, embed_size = val_tokens1.size()
    y_pred = sent_attn(val_tokens1,val_tokens2,  max_sents)
    return y_pred

# convert reviews to tokens
def tokenize_all_reviews(reviews_split):
    reviews_words = reviews_split.split(' ')
    tokenized_reviews = []
    for review in reviews_words:
        ints = []
        for word in review.split(' '):
            if(word==''):
                continue
            if(word=='.'):
                continue
            try:
                idx = embed_lookup.vocab[word].index
            except: 
                idx = 0
            tokenized_reviews.append(idx)
    return tokenized_reviews



def iterate_minibatches(inputs,  batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]

def pad_batch(mini_batch):
    mini_batch_size = len(mini_batch)
    max_sent_len = int(np.max([len(x) for x in mini_batch]))
    max_token_len = int(np.max([len(val) for sublist in mini_batch for val in sublist]))

    if(max_sent_len==1):
        max_sent_len=2
        
    if(max_sent_len >= 50):
        max_sent_len = 20
    if(max_token_len >= 300):
        max_token_len = 100
        
    main_matrix = np.zeros((mini_batch_size, max_sent_len, max_token_len), dtype= np.int)
    for i in range(main_matrix.shape[0]):
        for j in range(main_matrix.shape[1]):
            for k in range(main_matrix.shape[2]):
                try:
                    main_matrix[i,j,k] = mini_batch[i][j][k]
                except IndexError:
                    pass
    return Variable(torch.from_numpy(main_matrix).transpose(0,1))

def pad_batch2(mini_batch):
    mini_batch_size = len(mini_batch)
    max_sent_len = int(np.max([len(mini_batch[x]) for x in range(0,len(mini_batch))]))
    if(max_sent_len==1):
        max_sent_len=2
        
    if(max_sent_len >=300):
        max_sent_len = 100
        
    #print("max_sent_len:", max_sent_len)
    main_matrix = np.zeros((mini_batch_size, max_sent_len), dtype= np.int)
    for i in range(main_matrix.shape[0]):
         for k in range(main_matrix.shape[1]):
            try:
                main_matrix[i,k] = mini_batch[i][k]
            except IndexError:
                pass
    return Variable(torch.from_numpy(main_matrix))



def gen_minibatch2(tokens,  mini_batch_size, shuffle= True):
    for token in iterate_minibatches(tokens,  mini_batch_size, shuffle= shuffle):
        token1 = pad_batch(token)
        token_x=[]
        for i in token:
            c = []
            for j in i:
                for k in j:
                    c.append(k)
            token_x.append(c)

        token2 = pad_batch2(token_x)

        yield token1, token2 

def test_accuracy_full_batch3(a, mini_batch_size,  sent_attn):
    a_t=' '
    j=[]
    g=0
    tokens=list()
    labels=[]
    print("1111111")
    for e in a.split(' '):
        a_t +=e+' '

        c_t = list(tokenize_all_reviews(a_t))
        g+=1
        if(len(c_t) <=3):
            continue
        if('.' in e):
            b_t = list(tokenize_all_reviews(a_t))
            j.append(b_t)
            a_t=' '
        elif('?' in e):
            b_t = list(tokenize_all_reviews(a_t))
            j.append(b_t)
            a_t=' '
        elif('!' in e):
            b_t = list(tokenize_all_reviews(a_t))
            j.append(b_t)
            a_t=' '


    if(len(j) == 0 ):
        b_t = list(tokenize_all_reviews(a))
        j.append(b_t)
    elif(len(j) > 0 and len(a_t)>=3):
        b_t = list(tokenize_all_reviews(a_t))
        j.append(b_t)

    tokens.append(j)
    tokens.append(j)

    tokens = np.array(tokens)
    print(tokens)
    sent_attn.eval()

    p = []
    p2 = []
    l = []
   
    g = gen_minibatch2(tokens, mini_batch_size)

    for token1,token2 in g: 
        embedding = nn.Embedding.from_pretrained(weights)
        tokens1 = embedding(token1.long())
        tokens2 = embedding(token2.long())
        y_pred3 = get_predictions(tokens1,tokens2,  sent_attn)
        y_pred1=F.softmax(y_pred3)
        _, y_pred = y_pred1.max(1)

    return y_pred   




def train_early_stopping(mini_batch_size, X_train, y_train,X_dev,y_dev,
                          sent_attn,  sent_optimizer, loss_criterion, num_epoch ):
    start = time.time()
    loss_full = []
    
    
    epoch_counter = 0
    sent_attn.train()
    

    cur=0
    for i in range(1, num_epoch + 1):
        loss_epoch = []
        g = gen_minibatch2(X_train, y_train, mini_batch_size)
        for tokens1, tokens2, labels in g:
            embedding = nn.Embedding.from_pretrained(weights)
            tokens1 = embedding(tokens1.long())
            tokens2 = embedding(tokens2.long())
            loss = train_data(tokens1,tokens2, labels, sent_attn, 
                              sent_optimizer, loss_criterion)
            loss_full.append(loss.item())
            loss_epoch.append(loss.item())
    
        print ('Average training loss at this epoch..minibatch ' ,  np.mean(loss_epoch))
        vlos = check_val_loss(X_dev, y_dev, mini_batch_size, sent_attn)
        print( 'Test Loss at ',i, ' is ',vlos)
        cur2, acur,pre2, recall2 =test_accuracy_full_batch(X_dev, y_dev, 32,entireContext_model)
        print("f1-score(test) : ", cur2 ,  "accuracy :  ", acur, "recall :", recall2, "precision :", pre2)
        sent_attn.train()
        if(cur < cur2):
            cur = cur2
            print("saving...model")
            torch.save(sent_attn.state_dict(), 'abusive_detection33.pt')
            

                
            
        print("************************************************************************************************************")

        print ('Loss at %d minibatches, %d epoch,(%s) is %f' %(i, epoch_counter, timeSince(start), np.mean(loss_epoch)))


        print("************************************************************************************************************")   
            
   
    return 


def clean_text(temp):
    temp = temp.replace("rt :","")
    temp = temp.replace("rt:","")
    temp = temp.replace("rt","")
    temp = temp.replace("rt:","")
    temp = temp.replace("rt :","")
    temp = temp.replace(")","")
    
    temp = temp.replace("^",".")
    temp = temp.replace("~",".")
    temp = temp.replace("^",".")
    temp = temp.replace("~",".")
    temp = temp.replace("^",".")
    temp = temp.replace("~",".")
    
    temp = temp.replace("......",".")
    temp = temp.replace(".....",".")
    temp = temp.replace("....",".")
    temp = temp.replace("...",".")
    temp = temp.replace("..",".")
    
    temp = temp.replace("!!!!!!:","!")
    temp = temp.replace("!!!!!:","!")
    temp = temp.replace("!!!!:","!")
    temp = temp.replace("!!!:","!")
    temp = temp.replace("!!:","!")
    
    
    
    temp = temp.replace("??????:","?")
    temp = temp.replace("?????:","?")
    temp = temp.replace("????:","?")
    temp = temp.replace("???:","?")
    temp = temp.replace("??:","?")
    
    
    
    temp = temp.replace(". . . . . . .",".")
    temp = temp.replace(". . . . .",".")
    temp = temp.replace(". . . .",".")
    temp = temp.replace(". . .",".")
    temp = temp.replace(". .",".")
    
    
    
    temp = temp.replace("? ? ? ? ? ?","?")
    temp = temp.replace("? ? ? ? ?","?")
    temp = temp.replace("? ? ? ?","?")
    temp = temp.replace("? ? ?","?")
    temp = temp.replace("? ?","?")
    
    
    
    temp = temp.replace("! ! ! ! !","!")
    temp = temp.replace("! ! ! !","!")
    temp = temp.replace("! ! !","!")
    temp = temp.replace("! !","!")

    return temp


'''
Hierarchical C-LSTM 탐지 모델을 나타냄
self.net1은 문장 단위로 각 핵심 단어를 추출하기 위한 C-LSTM이고
self.net2는 문장 단위로 얻은 문맥 벡터를 통합하여 문맥 벡터를 추출하는 C-LSTM임
'''

class entireContext(nn.Module):
    
    def __init__(self):                
        
        super(entireContext, self).__init__()

        self.wordCLSTM = context_abusive.AttentionWordRNN(num_filters, kernel_sizes)
        self.senCLSTM = context_abusive.AttentionSentRNN(num_filters, kernel_sizes)
        self.Lin1 = nn.Linear(200,2)
 
    def forward(self, embed, source,max_sents):
        s = None
        for i in range(max_sents):
            _s = self.wordCLSTM(embed[i,:,:])
            if(s is None):
                s = _s
            else:
                s = torch.cat((s,_s),0)    

        y_pred1 = self.senCLSTM(s)
        y_pred = self.Lin1(y_pred1)
        return y_pred
    
   
embed_lookup = torch.load('./data/embed_lookup.pt')

weights = torch.load('./data/weights.pt')
final_attn = entireContext()
print("dsfasdfsfdasfasfa")

print_every = 100
vocab_size = 300000
output_size = 1 # binary class (1 or 0)
embedding_dim = 300
num_filters = 100
kernel_sizes = [3,4,5]
num_layer=1
num_classes =3
learning_rate = 0.001
momentum = 0.9
criterion = nn.CrossEntropyLoss()

entireContext_model = entireContext( )
entireContext_model.cuda()
entireContext_optimizer = torch.optim.Adam(list(entireContext_model.parameters()), lr=learning_rate)

kfold = KFold(n_splits=9,random_state=0,shuffle=True)



print("\nKFold**************")
aurrrrr = []
t=0
cur_a=1
c=0
print("###")

fr = open('./data/trainset.tsv', 'r', encoding='utf-8')
rdrr = csv.reader(fr,  delimiter='\t')

X_train=list()
y_train=list()
for line in rdrr:
    a = line[0]
    b = line[1]

    if '1' in b:
        b=int(b)
        b = 0
    elif '2' in b:
        continue
        b=int(b)
        b = 1
    elif '3' in b:
        b=int(b)
        b = 1
    else:
        continue
        
    if(a==''):
        continue
    a_t=' '
    j=[]

    for e in a.split(' '):
        a_t +=e+' '
        
        c_t = list(tokenize_all_reviews( a_t))
        
        if(len(c_t) <=3):
            continue
            
        if('.' in e):
            b_t = list(tokenize_all_reviews(a_t))
            j.append(b_t)
            a_t=' '
        elif('?' in e):
            b_t = list(tokenize_all_reviews(a_t))
            j.append(b_t)
            a_t=' '
        elif('!' in e):
            b_t = list(tokenize_all_reviews(a_t))
            j.append(b_t)
            a_t=' '
    if(len(j) == 0 ):
        b_t = list(tokenize_all_reviews(a))
        j.append(b_t)
    elif(len(j) > 0 and len(a_t)>=3):
        b_t = list(tokenize_all_reviews(a_t))
        j.append(b_t)


    if(len(j)==0):
        continue

    X_train.append(j)
    y_train.append(b)

fr.close()
X_train = np.array(X_train)
y_train = np.array(y_train)

print(len(y_train))

print(X_train.shape, y_train.shape)

for train_index, validate_index in kfold.split(X_train):
    print("train index:", train_index.shape, "validate index:", validate_index.shape)
    print(train_index)
    
    X_train1, X_dev = X_train[train_index], X_train[validate_index]
    y_train1, y_dev = y_train[train_index], y_train[validate_index]
    print(y_dev)
    print(X_dev)
    X_train1 = np.array(X_train1)
    y_train1 = np.array(y_train1)
    X_dev = np.array(X_dev)
    y_dev = np.array(y_dev)
    ac,b,c,d= train_early_stopping(64, X_train1, y_train1, X_dev,y_dev,entireContext_model, 
                                    entireContext_optimizer, criterion,  2000)
    break

print("%%%%%")
print(ac)
print(b)
print(c)
print(d)

print("%%%")



if __name__ == '__main__':
   app.run(ipaddress, debug = True, threaded = True)
