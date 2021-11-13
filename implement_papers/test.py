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
select_model = 'roberta-base_B=16_lr=5e-06_12_11_2021_08:05.pth'
PATH = '../../models/'+ select_model

checkpoint = torch.load(PATH,map_location=run_on)






