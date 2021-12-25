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
from loss import norm_vect 
from loss import intent_loss
from read_config import read_file_config 
import pprint


"""
Todo:

1. proof that sum each class before concate and concate first sum later 
got the same loss

"""

def compute_sim(h,labels): 
   
    skips = [] 
    idxs = np.arange(len(labels))

    print("idxs samples :",idxs)
    # masking each i  

    for idx in range(len(labels)):

        # select yi = yj

        if idx in skips:
            continue
        

        label_np = np.array(labels)


        mask = label_np[idx] == label_np

        mask[idx] = False



        if np.count_nonzero(mask)>=1:
            print(">>>>") 
            print("idx :",idx)
            print("current label :",label_np[idx])
            print("mask :",mask)

            print("current skiping :",idxs[mask])
            for val in idxs[mask]:

                if val not in skips:
                    skips.append(val)
            
            print("label i pairs :",label_np[mask])
            print(">>>")


    print("all skips :",skips)





path_finetuning = 'config/config3.yaml'
yaml_data = read_file_config(path=path_finetuning)

data = combine('CLINC150','train_5') 
train_examples = data.get_examples()

sampled_tasks = sample(yaml_data["training_params"]["N"], examples=train_examples,train=False) 

label_distribution, label_map = get_label_dist(sampled_tasks,train_examples,train=False)

labels = ['todo_list', 'routing','routing','routing','order_status', 'order_status', 'order_status','routing']

num_sam = len(labels) 

print("#labels",num_sam)

h  = torch.randn(num_sam,768)

print("h shape:",h.shape)

T, h_i, h_j = create_supervised_pair(h,labels,debug=False)

loss_s_cl = supervised_contrasive_loss(h_i, h_j, h, T,yaml_data["training_params"]["temp"],debug=False) 


print("#pairs :",T)
print("h_i:",h_i.shape)
print("h_j:",h_j.shape)
print("foward with function :",loss_s_cl)


print("------------------")
print("in func :",compute_sim(h,labels=labels))





