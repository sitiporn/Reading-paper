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
batch_size = 32 
labels = []
samples = []
epochs = 15 
lamda = 1.0
running_times = 10
lr=1e-5
model_name='roberta-base'

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
embedding = SimCSE(device='cuda:1',model_name=model_name) 
sim = Similarity(temperature)
train_loader = SenLoader(sampled_tasks)
data  = train_loader.get_data()

for i in range(len(data)):
   samples.append(data[i].text_a)
   labels.append(data[i].label)

optimizer= AdamW(embedding.parameters(), lr=lr)
train_data = CustomTextDataset(labels,samples)  
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)

print("DataLoader Done !")


