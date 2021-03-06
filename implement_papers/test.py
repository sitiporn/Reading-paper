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
#from loss import intent_classification_loss
from loss import norm_vect 
import yaml 
import json
import re

# get time 
now = datetime.now()
dt_str = now.strftime("%d_%m_%Y_%H:%M")

# config using yaml file 
with open('config/test.yaml') as file:

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

# 'roberta-base_B=16_lr=5e-06_27_11_2021_07:20.pth'
#'roberta-base_B=16_lr=0.001_27_11_2021_17:06.pth'

select_model = 'Load=True_roberta-base_B=16_lr=5e-07_22_12_2021_13:07.pth'
 
#"Load=True_roberta-base_B=16_lr=5e-06_18_12_2021_10:05.pth"
#"Load=True_roberta-base_B=16_lr=5e-06_18_12_2021_09:57.pth"

#'Load=True_roberta-base_B=16_lr=5e-06_13_12_2021_14:55.pth'



embedding = SimCSE(device=yaml_data["testing_params"]["device"],classify=yaml_data["model_params"]["classify"],model_name=yaml_data["model_params"]["model"]) 

embedding.load_model(select_model=select_model,strict=True)

# loading model 
# from config
"""
select_model = 'roberta-base_B=16_lr=5e-06_17_11_2021_09:51.pth'
PATH = '../../models/'+ select_model
""" 

"""
checkpoint = torch.load(PATH,map_location=yaml_data["testing_params"]["device"])
embedding.load_state_dict(checkpoint,strict=False)
"""

# Tensorboard 
logger = Log(load_weight=True,lamb=0.05,temp=0.5,experiment_name=yaml_data["model_params"]["exp_name"],model_name=yaml_data["model_params"]["model"],batch_size=yaml_data["testing_params"]["batch_size"],lr=yaml_data["testing_params"]["lr"])

# get test set  

data = combine('CLINC150','test')

print("Combine dataset done !:",len(data.get_examples()))

# load all datasets 
test_examples = data.get_examples()

#sampled_tasks = [sample(yaml_data["testing_params"]["N"], test_examples) for i in range(yaml_data["testing_params"]["T"])]
sampled_tasks = sample(yaml_data["testing_params"]["N"],examples=test_examples,train=False)
#print("len of examples",len(sampled_tasks[0]))

sim = Similarity()
test_loader = SenLoader(sampled_tasks)
#data  = test_loader.get_data()

samples = []
labels = [] 

#for i in range(len(data)):
for i in range(len(test_examples)):
   #samples.append(data[i].text_a)
   #labels.append(data[i].label)
   samples.append(test_examples[i].text)
   labels.append(test_examples[i].label)

optimizer= AdamW(embedding.parameters(), lr=yaml_data["testing_params"]["lr"])
test_data = CustomTextDataset(labels,samples,batch_size=yaml_data["testing_params"]["batch_size"])  
test_loader = DataLoader(test_data,batch_size=yaml_data["testing_params"]["batch_size"],shuffle=False,num_workers=2)

print("Test of Dataloader !")
label_distribution, label_maps = get_label_dist(sampled_tasks,test_examples,train=False)

print("label maps :",label_maps)

correct = 0
total = 0 
""""
bugs :

- mapping class(text) to label_ids is of model when we tune and test are not the same ids
"""

with torch.no_grad():
    for (idx, batch) in enumerate(test_loader): 
        
            _, outputs = embedding.encode(batch['Text'],batch['Class'],label_maps=label_maps,masking=False,train=False)

            logits = outputs.logits
            logits_soft = torch.softmax(logits,dim=-1)

            _, predicted = torch.max(logits_soft,dim=-1)

            if label_maps is not None: 
                
                labels = [label_maps[stringtoId] for stringtoId in (batch['Class'])]
                labels = torch.tensor(labels).unsqueeze(0)               
                total += labels.size(0)
                labels = labels.to(yaml_data["testing_params"]["device"])
                correct += (predicted == labels).sum().item()
            
            if debug:
                print(">>>>>>>>>>")
                print("labels: ",labels)
                print("class :",batch['Class'])
                print("<<<<<<<<<<")

                print("Predicted:",predicted)
                print("Correct :",correct)
                print("Total :",total)

                break

"""
print('Accuracy of the network : %d %%' % (
    100 * correct / total))

"""
