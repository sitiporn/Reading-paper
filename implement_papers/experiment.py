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



# get time 
now = datetime.now()
dt_str = now.strftime("%d_%m_%Y_%H:%M")

# config
N = 5  # number of samples per class (100 full-shot)
T = 1 # number of Trials
temperature = 0.1
batch_size = 16  
labels = []
samples = []
epochs = 30 
lamda = 1.0
running_times = 10
lr=1e-5
model_name='roberta-base'
run_on = 'cuda:1'
coeff = 0.7
running_time = 0.0
classify = True 


embedding = SimCSE(device=run_on,classify=classify,model_name=model_name) 
# loading model 
select_model = 'roberta-base_epoch14_B=16_lr=5e-06_01_11_2021_17:17.pth'
PATH = '../../models/'+ select_model
checkpoint = torch.load(PATH,map_location=run_on)
#print(checkpoint.keys())
#print(dir(checkpoint))
embedding.load_state_dict(checkpoint,strict=False)
print("Loading Pretain Model done!")

s1 = 'that would be yes' # yes
s2 = 'that is not false' # yes 
s3 = 'i would like to know who programmed this ai' # who_made_you 

u = [s1,s2,s3]

h1, _ = embedding.encode(u)

"""
-Roberata based 
   L Pretrain model(Encoder)  could not maximize sim even same class and lower sim on diff class 
-Bert based
   
"""

sim = Similarity()
pos1 = sim(h1[0,0,:],h1[0,0,:])
neg1 = sim(h1[0,0,:],h1[1,0,:])
neg2 = sim(h1[0,0,:],h1[2,0,:]) 

print("h1 :",h1.shape)
print("h[0,:,:] :",h1[0,:,:].shape)
print("shape:",pos1.shape)

print("Sim of the same uterance  pair:",pos1)
print("Sim of the same class of uterance :",neg1)
print("Sim of neg pair:",neg2)
print(embedding.eval())
