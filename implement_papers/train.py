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

valid_labels = []
valid_samples = []

epochs = 15 
lamda = 1.0
running_times = 10
lr=5e-6
model_name= "roberta-base" 
prior_weight = True 
train_file_path = '../../datasets/Few-Shot-Intent-Detection/Datasets/CLINC150/train/'

valid_file_path = '../../datasets/Few-Shot-Intent-Detection/Datasets/CLINC150/valid/'


# Tensorboard
logger = Log(load_weight=prior_weight,lamb=1.0,temp=temperature,experiment_name='Pretrain',model_name=model_name,batch_size=batch_size,lr=lr)

# combine all dataset
data = combine(exp_name='train') 
valid = combine(exp_name='valid')
print("Combine dataset done !:",len(data.get_examples()))
print("Combine validation !:",len(valid.get_examples()))

# load all datasets 
train_examples = data.get_examples()
valid_examples = valid.get_examples()
"""
structure of this data  [trials] 
trail -> [dict1,dict2,...,dict#intents]
every dict -> {'task':'lable name','examples':[text1,text2,..,textN]}
"""
sampled_tasks = [sample(N, train_examples) for i in range(T)]
valid_tasks = [sample(N,valid_examples) for i in range(T)]

print("len of examples",len(sampled_tasks[0]))
print("len of validation",len(valid_tasks[0]))


embedding = SimCSE(device='cuda:2',pretrain=prior_weight,model_name=model_name) 
sim = Similarity(temperature)

train_loader = SenLoader(sampled_tasks)
valid_loader = SenLoader(valid_tasks)

data  = train_loader.get_data()
valid_data = valid_loader.get_data()


# get samples and label from data
for i in range(len(data)):
   samples.append(data[i].text_a)
   labels.append(data[i].label)


for j in range(len(valid_data)):
   valid_samples.append(valid_data[j].text_a)
   valid_labels.append(valid_data[j].label)

optimizer= AdamW(embedding.parameters(), lr=lr)
train_data = CustomTextDataset(labels,samples)  
valid_data = CustomTextDataset(valid_labels,valid_samples)

train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=8)

valid_loader = DataLoader(valid_data,batch_size=batch_size,num_workers=8)



print("DataLoader Done !")


running_time = 0

for epoch in range(epochs):
    
    running_loss = 0.0
    running_loss_1 = 0.0
    running_loss_2 = 0.0

    for (idx, batch) in enumerate(train_loader):

        optimizer.zero_grad()
        
        # foward 2 times
        # get h_i
        h, _ = embedding.encode(sentence=batch['Text'])
        # get h_bar with random mask 0.10 among sentence in the batch
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
            #print('[%d, %5d] loss_total: %.3f loss_contrasive:  %.3f loss_language: %.3f ' %(epoch+1,idx+1,running_loss/running_times,running_loss_1/running_times,running_loss_2/running_times))

            print('[%d, %5d] loss_total: %.3f' %(epoch+1,idx+1,running_loss/running_times))
            logger.logging("Loss/train",(running_loss/running_times),running_time)
            running_loss = 0.0

            model = embedding.get_model()  

    
    valid_loss = 0.0      
    correct = 0
    total = 0

    for (idx, batch) in enumerate(valid_loader):
        
        # foward 2 times  
        h, _ = embedding.encode(sentence=batch['Text'],train=False)
        h_bar, outputs = embedding.encode(batch['Text'],debug=False,masking=True,train=False)
        

       
        hj_bar = create_pair_sample(h_bar,debug=False)    
        hj_bar = [ torch.as_tensor(tensor) for tensor in hj_bar]
        hj_bar = torch.stack(hj_bar)
        
        h_3d = h.unsqueeze(1)
        # print(hj_bar.shape,h_bar3d.shape)
        # print(len(hj_bar),len(hj_bar[0]),len(hj_bar[0][0]))
        # change it to be to taking grad
        
        loss_cl = contrasive_loss(h,h_bar,hj_bar,h_3d,temperature,batch_size) 
        loss_lml = outputs.loss

        #(batch_size,seq_len,vocab_size) 
        prediction = outputs.logits 
       
        """
        Todo

        1. select only mask token
        2. find softmax along that token
        3. find the highest prob of mask token   
        4. compare the label belong to mask pos with predict of mask
        """

        labels, mask_arr  = embedding.get_label()

        """
        prediction shape before mask: (batch_size,seq_len,vocab_size)
        prediction after masking : (#of masking,vocab_size)
        labels after masking : (#of masking)
        """
        """
        print("Prediction shape:",prediction.shape) 
        print("Predicton:",prediction[mask_arr])
        print("labels :",labels[mask_arr])
        """ 
        prediction = prediction[mask_arr]
        labels = labels[mask_arr]
       
        #print("prediction after masking ",prediction.shape)
        #print("labels after masking :",labels.shape)
        
        prediction = torch.softmax(prediction,dim=-1)
        prediction = torch.max(prediction,dim=-1)[1]

       
        correct += (prediction==labels).sum().item()          
        
        print("prediction:",prediction)
        print("labels :",labels)
        total += labels.size(0)
        print("correct :",correct)
        print("total :",total)

        valid_loss = loss_cl + (lamda*loss_lml)


    valid_acc =  100 * (correct/total)   
    print('Acc of validation: %d'%(valid_acc))
    logger.logging('Loss/Train',loss_stage1,epoch)
    logger.logging('Loss/validate',valid_loss,epoch)
    logger.logging('Accuracy/validation',valid_acc,epoch)
    print("Logging each epochs:")
         

      
#    for data, labels in validloader:

    PATH_to_save = f'../../models/{model_name}_epoch{epoch}_B={batch_size}_lr={lr}_{dt_str}.pth'
    torch.save(model.state_dict(),PATH_to_save)    
    print("save !:",PATH_to_save)


print('Finished Training')
torch.save(model.state_dict(),PATH_to_save)
print("Saving Done !")


        


   

