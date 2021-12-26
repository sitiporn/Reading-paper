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

def compute_sim(h,labels:list,temp:float): 
   
    T = 0 # the number of pairs
    skips = [] 
    loss_s_cl = [] 
    sim = Similarity(temp=temp)
    # idexign all samples
    idxs = np.arange(len(labels))

    

    print("idxs samples :",idxs)
    # masking each i  

    for idx in range(len(labels)):

        # select yi = yj

        if idx in skips:
            continue
        

        label_np = np.array(labels)


        # create mask for each i
        mask = label_np[idx] == label_np

        mask[idx] = False



        if np.count_nonzero(mask)>=1:
            
            print(">>>>") 
            print("idx :",idx)
            print("current label :",label_np[idx])
            print("mask :",mask)

            print("current skiping :",idxs[mask])

            h_i = h[idx,:]
            h_j = h[mask,:] 

            T += np.count_nonzero(mask)
            print("# pairs cumulative :",T)
            h_i_broad = h_i.repeat(np.count_nonzero(mask),1) 

            print("after broadcast :",h_i_broad.shape)
            print("h_j :",h_j.shape)

            res_i = torch.exp(sim(h_i,h_j))
            
            print("res_i :",res_i.shape)

            res_i = torch.sum(res_i,dim=0)


            for val in idxs[mask]:

                if val not in skips:
                    skips.append(val)
            
            print("label i pairs :",label_np[mask])

            h_i_bot =   h_i.repeat(h.shape[0],1)
        
            print("broadcast bot:",h_i_bot.shape)
            print("h_n :",h.shape)

            bot = torch.exp(sim(h_i_bot,h))
            print("compute bot before sum:",bot.shape)

            bot = torch.sum(bot,dim=0)
            print("bottom :",bot.shape)

            loss_i  = torch.log(res_i / bot)

            loss_i = torch.sum(loss_i,dim=0)

            loss_s_cl.append(loss_i)


            print(">>>")

    # convert list of tensor 
    contrasive_loss = torch.sum(torch.Tensor(loss_s_cl),dim=0)* (-1/T)
    # combine all loss_i
    print("all skips :",skips)

    #

    return contrasive_loss 



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


print("------------------")
print("compare func :",compute_sim(h,labels=labels,temp=yaml_data["training_params"]["temp"]))
print("original func :",loss_s_cl)





