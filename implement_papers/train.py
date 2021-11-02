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
# Todo: making a batch that should be able to train 
# 1.) feed all data in batch twice through encoder
# 2.) create lstage1 = self_cl_loss + lamda * mlm_loss


# get time 
now = datetime.now()
dt_str = now.strftime("%d_%m_%Y_%H:%M")

# config
N = 100  # number of samples per class (100 full-shot)
T = 1 # number of Trials
temperature = 0.1
batch_size = 16
labels = []
samples = []
epochs = 15 
lamda = 1.0
running_times = 10
lr=5e-6
model_name= "roberta-base" 

train_file_path = '../../datasets/Few-Shot-Intent-Detection/Datasets/CLINC150/train/'

# Tensorboard
logger = Log(experiment_name='Pretrain',model_name=model_name,batch_size=batch_size,lr=lr)

# combine all dataset
data = combine() 
print("Combine dataset done !:",len(data.get_examples()))

# load all datasets 
train_examples = data.get_examples()
"""
structure of this data  [trials] 
trail -> [dict1,dict2,...,dict#intents]
every dict -> {'task':'lable name','examples':[text1,text2,..,textN]}
"""
sampled_tasks = [sample(N, train_examples) for i in range(T)]
print("len of examples",len(sampled_tasks[0]))
embedding = SimCSE(device='cuda:2',model_name=model_name) 
sim = Similarity(temperature)
train_loader = SenLoader(sampled_tasks)
data  = train_loader.get_data()

for i in range(len(data)):
   samples.append(data[i].text_a)
   labels.append(data[i].label)

optimizer= AdamW(embedding.parameters(), lr=lr)
train_data = CustomTextDataset(labels,samples)  
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=8)

print("DataLoader Done !")


running_time = 0

for epoch in range(epochs):
    
    running_loss = 0.0
    running_loss_1 = 0.0
    running_loss_2 = 0.0

    for (idx, batch) in enumerate(train_loader):

        optimizer.zero_grad()
        
        # foward
        h, _ = embedding.encode(batch['Text'])
        h_bar, outputs = embedding.encode(batch['Text'],debug=False,masking=True)
        hj_bar = create_pair_sample(h_bar,debug=False)    
        hj_bar = [ torch.as_tensor(tensor) for tensor in hj_bar]
        hj_bar = torch.stack(hj_bar)
        
        h_3d = h.unsqueeze(1)
        # print(hj_bar.shape,h_bar3d.shape)
        # print(len(hj_bar),len(hj_bar[0]),len(hj_bar[0][0]))
        # change it to be to taking grad
        
        loss_cl = contrasive_loss(h,h_bar,hj_bar,h_3d,temperature,batch_size) 
        loss_lml = outputs.loss

        # loss of pretrain model 
        loss_stage1 = loss_cl + (lamda*loss_lml)
        
        loss_stage1.backward()
        optimizer.step()

        # print statistics
        running_loss += loss_stage1.item()
       # running_loss_1 += loss_cl.item()
       # running_loss_2 += loss_lml.item()
        torch.cuda.empty_cache() 
         
        if idx % running_times == running_times-1: # print every 50 mini-batches
            running_time += 1
            logger.logging('Loss/Train',running_loss,running_time)
            #print('[%d, %5d] loss_total: %.3f loss_contrasive:  %.3f loss_language: %.3f ' %(epoch+1,idx+1,running_loss/running_times,running_loss_1/running_times,running_loss_2/running_times))
            print('[%d, %5d] loss_total: %.3f' %(epoch+1,idx+1,running_loss/running_times))
            running_loss = 0.0
            logger.close()
            model = embedding.get_model()  

        
    
    PATH_to_save = f'../../models/{model_name}_epoch{epoch}_B={batch_size}_lr={lr}_{dt_str}.pth'
    torch.save(model.state_dict(),PATH_to_save)    
    print("save !:",PATH_to_save)


print('Finished Training')
torch.save(model.state_dict(),PATH_to_save)
print("Saving Done !")


        

"""
sentence = ["get an uber to take me to my brother's house in mineola",'i need to transfer from this account to that one']

embed = embedding.encode(sentence,debug=False)
print(embed)
#print(dir(training_data))

print("Training:",embed)
# Note : same intent sim higher than different intents
# but the diff one not quite well yet
print(sim(embed[0],embed[1]))
print(sim(embed[0],embed[9]))

"""
   

