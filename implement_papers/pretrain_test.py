import torch
import urllib
import os 
from torch.utils.tensorboard import SummaryWriter 
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from datetime import datetime
from utils import loss
from dataloader import IntentExample
from dataloader import load_intent_examples
from dataloader import sample
from dataloader import InputExample
from loss import SimCSE
from loss import Similarity
from dataloader import SenLoader 
from dataloader import CustomTextDataset
from torch.utils.data import Dataset, DataLoader
from dataloader import create_pair_sample
from loss import contrasive_loss
from transformers import AdamW
from torch.autograd import Variable
from logger import Log
from dataloader import combine
from dataloader import create_supervised_pair
from loss import supervised_contrasive_loss 
from loss import get_label_dist
from loss import intent_classification_loss
from loss import norm_vect 
import yaml 
import json
import random


# get time 
now = datetime.now()
dt_str = now.strftime("%d_%m_%Y_%H:%M")

# config
N = 100  # number of samples per class (100 full-shot)
T = 1 # number of Trials
temperature = 0.1
batch_size = 16

test_labels = []
test_samples = []

valid_labels = []
valid_samples = []

epochs = 15 
lamda = 1.0
running_times = 10
lr=5e-6
model_name= "roberta-base" 
pretrain = False 


# from config
select_model = 'Load=False_roberta-base_epoch14_B=16_lr=5e-06_08_12_2021_07:21.pth'

#roberta-base_epoch14_B=16_lr=5e-06_25_11_2021_12:07.pth


# embedding 
embedding = SimCSE(device='cuda:2',pretrain=pretrain,model_name=model_name)
embedding.load_model(select_model=select_model)

# combine all datasets


valid = combine(exp_name='valid')
test = combine(exp_name='test')

print("Combine valid sets !",len(valid.get_examples()))
print("combine test sets !",len(test.get_examples()))

# load all datasets
valid_examples = valid.get_examples()
test_examples = test.get_examples()


valid_task = [sample(N,valid_examples) for i in range(T)]
test_task = [sample(N,test_examples) for i in range(T)]

print("len of valid examples",len(valid_task[0]))
print("len of test examples",len(test_task[0]))

"""
what it might be wrong 

1. validation during train it's good but when we load the model to eval they cannot predict 
at least one correction -> something wrong with loading process  
"""

sim = Similarity(temperature)

valid_loader = SenLoader(valid_task)
test_loader = SenLoader(test_task)


valid = valid_loader.get_data()
test = test_loader.get_data()


# get data and label from valid data 
for i in range(len(valid)):
   valid_samples.append(valid[i].text_a) 
   valid_labels.append(valid[i].label)


for j in range(len(test)):
   test_samples.append(test[i].text_a)
   test_labels.append(test[i].label)


valid_data = CustomTextDataset(valid_labels,valid_samples,batch_size=batch_size)
test_data = CustomTextDataset(test_labels,test_samples,batch_size=batch_size)

valid_loader = DataLoader(valid_data,batch_size=batch_size,num_workers=8)
test_loader = DataLoader(test_data,batch_size=batch_size,num_workers=8)




correct = 0
total = 0

for (idx, batch) in enumerate(valid_loader):

   #foward 2 times
   h, _ = embedding.encode(sentence=batch['Text'],train=False)
   h_bar, outputs = embedding.encode(batch['Text'],debug=False,masking=True,train=False)

   
   # https://stackoverflow.com/questions/63040954/how-to-extract-and-use-bert-encodings-of-sentences-for-text-similarity-among-sen 
   # use value of CLS token 
   # select first token  
   h = h[:,0,:]
   h_bar = h_bar[:,0,:]

   # get negative pairs : (batch_size,batch_size-1,embed_size) 
   hj_bar = create_pair_sample(h_bar,debug=False)
   hj_bar = [torch.as_tensor(tensor) for tensor in hj_bar]
   hj_bar = torch.stack(hj_bar)
   
   # convert h -> h3d tensor 
   h_3d = h.unsqueeze(1)
   """
   hi: torch.Size([16, 768]) 
   h_bar: torch.Size([16, 768])
   hj_bar: torch.Size([16, 15, 768])
   hi_3d : torch.Size([16, 1, 768])
      
   print("shape after")  
   print(h.shape)       
   print(h_bar.shape)
   print(hj_bar.shape)
   print(h_3d.shape)
   
   """ 
     
   
   pos_sim, neg_sim, _  = contrasive_loss(h,h_bar,hj_bar,h_3d,temperature,batch_size)
   
   print("pos_sim:",pos_sim)
   print("neg_sim:",neg_sim)

   prediction = outputs.logits

   label, mask_arr = embedding.get_label()
   
   prediction = prediction[mask_arr]
   labels = label[mask_arr]

   prediction = torch.softmax(prediction,dim=-1)
   prediction = torch.max(prediction,dim=-1)[1]

   
   correct += (prediction==labels).sum().item()
   total += labels.size(0)
   
   print("Positive similarity",pos_sim)
   print("Negative similarity",neg_sim)
   print("correct :",correct)
   print("total :",total)

acc_valid = 100 * (correct / total)
"""
correct = 0
total = 0

for (idx, batch) in enumerate(test_loader):

   #foward 2 times
   h, _ = embedding.encode(sentence=batch['Text'],train=False)
   h_bar, outputs = embedding.encode(batch['Text'],debug=False,masking=True,train=False)

   hj_bar = create_pair_sample(h_bar,debug=False)
   hj_bar = [torch.as_tensor(tensor) for tensor in hj_bar]
   hj_bar = torch.stack(hj_bar)
           
   h_3d = h.unsqueeze(1)

   pos_sim, neg_sim, _ = contrasive_loss(h[:,0,:],h_bar[:,0,:],hj_bar[:,0,:],h_3d[:,0,:],temperature,batch_size)

   prediction = outputs.logits

   label, mask_arr = embedding.get_label()
   
   prediction = prediction[mask_arr]
   labels = label[mask_arr]

   prediction = torch.softmax(prediction,dim=-1)
   prediction = torch.max(prediction,dim=-1)[1]


   print("prediction:",prediction)
   print("labels :",labels)

   correct += (prediction==labels).sum().item()
   total += labels.size(0)
   print("Positive similarity",pos_sim)
   print("Negative similarity",neg_sim)
   print("correct :",correct)
   print("total :",total)


acc_test = 100 * (correct/total)
"""
print('Accuracy of valid: %d %%' % (acc_valid))
#print('Accuracy of test: %d %%' %(acc_test))

