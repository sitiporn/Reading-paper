import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import AutoModel, AutoTokenizer
import random
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray
from typing import List, Dict, Tuple, Type, Union
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np

class IntentExample(object):

    def __init__(self, text, label, do_lower_case):
        self.original_text = text
        self.text = text
        self.label = label

        if do_lower_case:
            self.text = self.text.lower()


class InputExample(object):

    def __init__(self, text_a, text_b, label = None):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        
def load_intent_examples(file_path, do_lower_case=True):
  
    examples = []

    with open('{}/seq.in'.format(file_path), 'r', encoding="utf-8") as f_text, open('{}/label'.format(file_path), 'r', encoding="utf-8") as f_label:
        for text, label in zip(f_text, f_label):
            e = IntentExample(text.strip(), label.strip(), do_lower_case)
            examples.append(e)

    return examples

def sample(N, examples):
    labels = {} # unique classes
    
    # check whether example's labels already exist or not 
     
    for e in examples:
        # if they have label in list append sample into the class 
        if e.label in labels:
            labels[e.label].append(e.text)
        # if they dont have lable create new one
        else:
            labels[e.label] = [e.text]
    
    sampled_examples = []
    # lables -> the number of intents
    # loop all labels
    for l in labels:
        # shuffle text in each label
        random.shuffle(labels[l])
        ## just slicing the numbers of sample
        ## in each label
        if l == 'oos':
            examples = labels[l][:N]
        else:
            examples = labels[l][:N]
        # each intent would be appendend N examples 
        sampled_examples.append({'task': l, 'examples': examples})

    return sampled_examples


class SenLoader(Dataset):
    def __init__(self,sentence,T:int=1):

        #params
        self.label_list = []
        self.intent_examples = []
        # number of trials
        self.T = T
        self.sample_task = sentence

    def get_data(self,trial:int=0):

        for idx in range(self.T):

            tasks = self.sample_task[idx]
            #print(type(tasks))
            self.label_list.append([])
            self.intent_examples.append([])
            
            for task in tasks:
                 label= task['task']
                 #print(label)
                 examples = task['examples']
                 self.label_list[-1].append(label)

                 for j in range(len(examples)):
                     
                     self.intent_examples[-1].append(InputExample(examples[j],None, label))
        return self.intent_examples[trial]

# create custom dataset class
class CustomTextDataset(Dataset):
    def __init__(self,labels,text):
        self.labels = labels
        self.text = text

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.text[idx]
        sample = {"Class": label,"Text": data}
    
        return sample

def create_pair_sample(h_bar,debug:bool=False):
    """
    h ->  a , b, c, a
    h'->  a', b',c',a'
    intent_idx -> (0,1,2,0)
    h   ->      [a, b, c, a ] for all i batch
    h_pos_bar = [a',b',c',a'] for all i batch 
    h_neg_bar = [[b',c',a'],[a',c',a'],[a',b',a'],[a',b',c']] 
              
    """
    h_neg_bar = []

    for idx in range(h_bar.shape[0]):
        
        mask = np.arange(h_bar.shape[0])
       
        if debug == True:
            print("===== Masking neg samples")
            print(mask.shape)
        #select neg pair hj in papers Simcse
        masking = (mask != idx)
        
        if debug == True:
            print(masking)
            print(h_bar[masking,:].shape)
            print("checking slice masking:")
            print(h_bar[masking,:3])
            print("check hidden bar without slicing")
            print(h_bar[:,:3])

        h_neg_bar.append(h_bar[masking,:])
        
    return h_neg_bar

class combine:

    def __init__(self):
      
        #params
        self.datasets = ['ATIS','BANKING77','CLINC150','HWU64','SNIPS']
        
        
    def get_examples(self): 

        combine = [] 
        for data in self.datasets:
            train_file_path = f'../../datasets/Few-Shot-Intent-Detection/Datasets/{data}/train/'
            train_examples = load_intent_examples(train_file_path)
            print(len(train_examples))
            print(type(train_examples))
            combine.append(train_examples)

        
        
        print(len(combine))
        print(len(combine[0]))
        print(len(combine[1]))         
        print(len(combine[2]))         
        print(len(combine[3]))         
        flat_combine_list = [item for sublist in combine for item in sublist]
        assert len(flat_list) == len(combine[0]) + len(combine[1]) + len(combine[2]) + len(combine[3]) + len(combine[4])
        print(len(flat_list))

        return flat_combine_list
















