import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np

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
# Todo: making a batch that should be able to train 
# 1.) feed all data in batch twice through encoder
# 2.) create lstage1 = self_cl_loss + lamda * mlm_loss

# config

train_file_path = '../../dataset/Few-Shot-Intent-Detection/Datasets/CLINC150/train/'
N = 100  # number of samples per class (100 full-shot)
T = 1 # number of Trials
temperature = 0.1

# 
train_examples = load_intent_examples(train_file_path)
"""
structure of this data  [trials] 
trail -> [dict1,dict2,...,dict#intents]
every dict -> {'task':'lable name','examples':[text1,text2,..,textN]}
"""
sampled_tasks = [sample(N, train_examples) for i in range(T)]


#print(sampled_tasks[0][0])
embedding = SimCSE('cuda') 
sim = Similarity(temperature)
train_loader = SenLoader(sampled_tasks)
# Testing fist example 
pos_sentence = sampled_tasks[0][0]['examples']

neg_sentence = sampled_tasks[0][1]['examples']




"""y
#sentence = ["what's local slang for goodbye in hawaii"] 
sentence = [
    "There's a kid on a skateboard.",
    "A kid is skateboarding.",
    "A kid is inside the house."
]

"""

data  = train_loader.get_data()

print(data[0])
print(dir(data[0]))
labels = []
samples = []

for i in range(len(data)):
   samples.append(data[i].text_a)
   labels.append(data[i].label)

batch_size = 64 
train_data = CustomTextDataset(labels,samples)  
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
"""
 Todo : Trainning  
 1.) making dataloader X
 2.) iterate batch     X
 3.) create pos_example X 
 4.) create neg_example
 5.) calculate cost of each examples
 6.) calculate loss summarize    

"""
for (idx, batch) in enumerate(train_loader):

    # Print the 'text' data of the batch
    print(idx, 'data: ', batch, '\n')
    
    
    #print(dir(batch))
    # get hidden representation from ui
    # Todo : proof of emebedding
    h = embedding.encode(batch['Text'],debug=False)
    h_bar = embedding.encode(batch['Text'],debug=False,masking=False)
    print(h_bar.shape)
    hj_bar = create_pair_sample(h_bar,debug=True)    
    
    


    """
    print("h:",h)  
    print("h:",h.shape)
    print("h_bar",h_bar)
    print("h_bar",h_bar.shape)
    print(batch['Class'])
    """
    # create pos samples h , h_bar -> same class 
     
    #hi_bar = embedding


    break

 




"""
print("== 1 ==")
print(len(training_data[0]))
print("== 2 ==")
print(training_data[0][0])
print(dir(training_data[0][0]))

"""

"""
sentence = ["get an uber to take me to my brother's house in mineola",'i need to transfer from this account to that one']

embed = embedding.encode(sentence,debug=False)
print(embed)
#print(dir(training_data))

print("Training:",embed)
# Note : same intent sim higher than different intents
# but the diff one not quite well yet
print(sim(embed[0],embed[1]))
print(sim(embed[0],embed[9]))

"""
   

