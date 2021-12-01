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

labels = []
samples = []
batch_size = 16  

# get single dataset  
data = combine('CLINC150','train_5') 

# load all datasets 
train_examples = data.get_examples()

print("len of datasets :",len(train_examples))

# sample dataset 
sample_task = sample(yaml_data["training_params"]["N"],examples=train_examples,train=False)

print("len(sample_task):",len(sample_task))
#print(sample_task)
label_distribution, label_map = get_label_dist(sample_task,train_examples,train=True)


train_loader = SenLoader(sample_task)
#data = train_loader.get_data()


for i in range(len(train_examples)):
   samples.append(train_examples[i].text)
   labels.append(train_examples[i].label)

train_data = CustomTextDataset(labels,samples,batch_size=batch_size)  
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)

for i, batch in enumerate(train_loader):

    print("set batch :",len(set(batch['Class'])))
    print("len batch :",len(batch['Class']))

    if len(set(batch['Class'])) == len(batch['Class']):
        
        print("batch :")
        print(i,batch['Class'])
    
    """ 
    if i==5:
        break
    """

"""

#print("label_distribution:",label_distribution)



#print(embedding.eval())

"""



