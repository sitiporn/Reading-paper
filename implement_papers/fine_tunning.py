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
N = 10  # number of samples per class (100 full-shot)
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


embedding = SimCSE(device='cuda:2',model_name=model_name) 
# loading model 
select_model = 'roberta-base_epoch14_B=16_lr=5e-06_01_11_2021_17:17.pth'
PATH = '../../models/'+ select_model
checkpoint = torch.load(PATH)
#print(checkpoint.keys())
#print(dir(checkpoint))
embedding.load_state_dict(checkpoint,strict=False)
print("Loading Pretain Model done!")

#print(dir(embedding))
# Tensorboard
logger = Log(experiment_name='Pretrain',model_name=model_name,batch_size=batch_size,lr=lr)

# get single dataset  
data = combine('CLINC150') 

print("len of datasets! :",len(data.get_examples()))

# load all datasets 
train_examples = data.get_examples()

sampled_tasks = [sample(N, train_examples) for i in range(T)]
print("the numbers of intents",len(sampled_tasks[0]))
train_loader = SenLoader(sampled_tasks)
data = train_loader.get_data()

#print(embedding.eval())

for i in range(len(data)):
   samples.append(data[i].text_a)
   labels.append(data[i].label)

optimizer= AdamW(embedding.parameters(), lr=lr)
train_data = CustomTextDataset(labels,samples)  
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)

print("DataLoader Done !")

for epoch in range(epochs):

    running_loss = 0.0

    for (idx, batch) in enumerate(train_loader): 
    

        optimizer.zero_grad()
        print(len(batch['Class']))
        print(len(batch['Text']))
        # foward
        #h, _ = embedding.encode(batch['Text'])

        print("break")
        break
        """ 
        h_bar, outputs = embedding.encode(batch['Text'],debug=False,masking=True)
        hj_bar = create_pair_sample(h_bar,debug=False)    
        hj_bar = [ torch.as_tensor(tensor) for tensor in hj_bar]
        hj_bar = torch.stack(hj_bar)

        """
