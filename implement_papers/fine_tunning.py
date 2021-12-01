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
import ruamel.yaml
import json
import re 


# get time 
now = datetime.now()
dt_str = now.strftime("%d_%m_%Y_%H:%M")

# config using yaml file
with open('config/config.yaml') as file:

    yaml_data = yaml.safe_load(file)
    
    jAll = json.dumps(yaml_data)

    loader = yaml.SafeLoader

    loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

    yaml_data = yaml.load(jAll, Loader=loader) 

# search hyperparameters

temp = [0.1, 0.3, 0.5]
lamda = [0.01,0.03,0.05]

# daclare some variable

labels = []
samples = []
running_time = 0.0 



# loading model 
#select_model = 'roberta-base_epoch14_B=16_lr=5e-06_01_11_2021_17:17.pth'
#PATH = '../../models/'+ select_model
#checkpoint = torch.load(PATH,map_location=yaml_data["training_params"]["device"])

select_model = 'roberta-base_epoch14_B=16_lr=5e-06_25_11_2021_12:07.pth'


# create dummy model 
embedding = SimCSE(device=yaml_data["training_params"]["device"],classify=yaml_data["model_params"]["classify"],model_name=yaml_data["model_params"]["model"]) 

#embedding.load_state_dict(checkpoint,strict=False)
embedding.load_model(select_model=select_model,strict=False)
print("Loading Pretain Model done!")

print("Freeze Backboned layers")
embedding.freeze_layers()

# get single dataset  
data = combine('CLINC150','train_5') 


# load all datasets 
train_examples = data.get_examples()

print("len of datasets :",len(train_examples))

#sample datasets
sampled_tasks = sample(yaml_data["training_params"]["N"], examples=train_examples,train=False) 

print("the numbers of intents",len(sampled_tasks))

label_distribution, label_map = get_label_dist(sampled_tasks,train_examples,train=False)

#print("label_distribution:",label_distribution)

train_loader = SenLoader(sampled_tasks)

"""
#data = train_loader.get_data()

#print(embedding.eval())
"""

for i in range(len(train_examples)):
   samples.append(train_examples[i].text)
   labels.append(train_examples[i].label)



train_data = CustomTextDataset(labels,samples,batch_size=yaml_data["training_params"]["batch_size"])  

train_loader = DataLoader(train_data,batch_size=yaml_data["training_params"]["batch_size"],shuffle=True)

print("DataLoader Done !")


print("lamda :",yaml_data["training_params"]["lamda"])
print("temperature :",yaml_data["training_params"]["temp"])
 
        
# Tensorboard
logger = Log(load_weight=True,lamb=yaml_data["training_params"]["lamda"],temp=yaml_data["training_params"]["temp"],experiment_name=yaml_data["model_params"]["exp_name"],model_name=yaml_data["model_params"]["model"],batch_size=yaml_data["training_params"]["batch_size"],lr=yaml_data["training_params"]["lr"])




# create optimizer
optimizer= AdamW(embedding.parameters(), lr=yaml_data["training_params"]["lr"])

total = 0
skip_time = 0 


for epoch in range(yaml_data["training_params"]["n_epochs"]):

    running_loss = 0.0
    running_loss_s_cl = 0.0
    running_loss_intent = 0.0 

    for (idx, batch) in enumerate(train_loader): 
    

        optimizer.zero_grad()

        total +=1
         
        
        # (batch_size, seq_len, hidhen_dim) 
        if len(set(batch['Class'])) == len(batch['Class']):
            print("no positive pairs !")
      
        h, outputs = embedding.encode(batch['Text'],batch['Class'],label_maps=label_map,masking=False)
        
        # https://stackoverflow.com/questions/63040954/how-to-extract-and-use-bert-encodings-of-sentences-for-text-similarity-among-sen 
        # use value of CLS token 
        h = h[:,0,:]
        

        
        T, h_i, h_j = create_supervised_pair(h,batch['Class'],debug=False)
        # (batch_size, seq_len, vocab_size) 
        logits = outputs.logits
        
                        
        #loss_s_cl = 0.0
     
        if h_i is None:
            print("skip this batch")
            skip_time +=1
            continue


        loss_s_cl = supervised_contrasive_loss(h_i, h_j, h, T,yaml_data["training_params"]["temp"],debug=False) 

        label_ids, _  = embedding.get_label()
          
       
        loss_intent = intent_classification_loss(label_ids, logits, label_distribution, coeff=yaml_data["training_params"]["smoothness"], device=yaml_data["training_params"]["device"])

        running_loss_intent = loss_intent.item()         
        
        loss_stage2 = loss_s_cl + (yaml_data["training_params"]["lamda"] * loss_intent)
        
        

        loss_stage2.backward()
        optimizer.step()

        # collect for visualize 
        running_loss += loss_stage2.item()
        running_loss_intent += loss_intent.item() 
        running_loss_s_cl += loss_s_cl.item()
        
        if idx % yaml_data["training_params"]["running_times"] ==  yaml_data["training_params"]["running_times"]-1: # print every 50 mini-batches
            running_time += 1
            logger.logging('Loss/Train',running_loss,running_time)
            print('[%d, %5d] loss_total: %.3f loss_supervised_contrasive:  %.3f loss_intent :%.3f ' %(epoch+1,idx+1,running_loss/yaml_data["training_params"]["running_times"] ,running_loss_s_cl/yaml_data["training_params"]["running_times"] ,running_loss_intent/yaml_data["training_params"]["running_times"]))
            
            print("skip_time:",skip_time)
            print("total :",total)

            #print('[%d, %5d] loss_total: %.3f' %(epoch+1,idx+1,running_loss/running_times))
            running_loss = 0.0
            logger.close()
            model = embedding.get_model()   

    

del logger    
print("deleate logger for one combination")

PATH_to_save = f'../../models/{yaml_data["model_params"]["model"]}_B={yaml_data["training_params"]["batch_size"]}_lr={yaml_data["training_params"]["lr"]}_{dt_str}.pth'

print(PATH_to_save)

print('Finished Training')
torch.save(model.state_dict(),PATH_to_save)
print("Saving Done !")
