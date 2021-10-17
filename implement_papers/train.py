import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np

from utils import loss
from dataloader import IntentExample
from dataloader import load_intent_examples
from dataloader import sample
from dataloader import InputExample
from dataloader import SimCSE
from dataloader import Similarity
from dataloader import SenLoader 
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




"""
#sentence = ["what's local slang for goodbye in hawaii"] 
sentence = [
    "There's a kid on a skateboard.",
    "A kid is skateboarding.",
    "A kid is inside the house."
]

"""
training_data = train_loader.get_data()

print(dir(training_data))
#embed = embedding.encode(sentence)

"""
print("Training:",embed)
# Note : same intent sim higher than different intents
# but the diff one not quite well yet
print(sim(embed[0],embed[1]))
print(sim(embed[0],embed[9]))

"""
