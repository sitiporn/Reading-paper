import torch
import urllib
import os 
from torch.utils.tensorboard import SummaryWriter 
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from logger import Log

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
from datetime import datetime
import torch.optim as optim


#get time 
now = datetime.now()
dt_str = now.strftime("%d_%m_%Y_%H:%M")


# config
train_file_path = '../../datasets/Few-Shot-Intent-Detection/Datasets/CLINC150/train/'
valid_file_path = '../../datasets/Few-Shot-Intent-Detection/Datasets/CLINC150/valid/'

N = 100  # number of samples per class (100 full-shot)
T = 1 # number of Trials

temperature = 0.1
batch_size = 1#64 

labels = []
samples = []

valid_labels = []
valid_samples = []


epochs = 15 
lamda = 1.0
iters = 10
lr = 5e-6
model_name = "roberta-base"
prior_weight = False 


PATH_to_save = f'../../models/Load={prior_weight}_{model_name}_B={batch_size}_lr={lr}_{dt_str}.pth'

print("Path to save :", PATH_to_save)

# Tensorboard
logger = Log(load_weight=prior_weight,num_freeze=0,lamb=1.0,temp=temperature,experiment_name='Pretrain',model_name=model_name,batch_size=batch_size,lr=lr)
# load datasets 

train_examples = load_intent_examples(train_file_path)
"""
structure of this data  [trials] 
trail -> [dict1,dict2,...,dict#intents]
every dict -> {'task':'lable name','examples':[text1,text2,..,textN]}
"""
sampled_tasks = [sample(N, train_examples) for i in range(T)]

sim = Similarity(temperature)
train_loader = SenLoader(sampled_tasks)
data = train_loader.get_data()

for i in range(len(data)):
    samples.append(data[i].text_a)
    labels.append(data[i].label)

num_classes = len(np.unique(np.array(labels)))
print("=== the numbers of classes == :", num_classes)


embedding = SimCSE(device='cuda:0', num_class=num_classes, pretrain=False)

optimizer= optim.Adam(embedding.parameters(), lr = lr)
train_data = CustomTextDataset(labels, samples, batch_size=batch_size)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

"""
 Todo : Programming:  Training
  1) combined all datasets of intents
  2) drop sentence that less than one sentences
  3) Trainnig BERT from scatch -> pretrain
  4) create mask language loss
"""
running_time = 0

for epoch in range(epochs):
    
    running_loss = 0.0
    running_loss_1 = 0.0
    running_loss_2 = 0.0

    for (idx, batch) in enumerate(train_loader):

       
        optimizer.zero_grad()
        
        # foward 2 times

        #get h_i 
        h, _ = embedding.encode(batch['Text'],debug=False,masking=False)
        
        # get h_bar with random mask 0.10 among sentence in the batch
        h_bar, outputs = embedding.encode(batch['Text'], debug=False, masking=True)

        
        _,_, loss_cl = contrasive_loss(h=h,h_bar=h_bar,temp=temperature,N=batch_size,compute_loss=True,debug=False) 

        loss_lml = outputs.loss

        # loss of pretrain model 
        loss_stage1 = loss_cl + (lamda*loss_lml)
        
        loss_stage1.backward()
        optimizer.step()

        # print statistics
        running_loss += loss_stage1
        running_loss_1 += loss_cl
        running_loss_2 += (lamda * loss_lml)
        
        if idx % iters == iters-1: # print every 10 mini-batches
            running_time += 1

            print('[%d, %5d] loss_total: %.3f loss_contrasive:  %.3f loss_language: %.3f ' %(epoch+1,idx+1,running_loss/(iters+1),running_loss_1/(iters+1),running_loss_2/(iters+1)))



            logger.logging("Loss/train",running_loss/(iters+1),running_time)
            running_loss = 0.0
            running_loss_1 = 0.0 
            running_loss_2 = 0.0


        
model = embedding.get_model()  
print('Finished Training')
print("path to save :",PATH_to_save)
torch.save(model.state_dict(),PATH_to_save)
print("Saving Done !")

   

