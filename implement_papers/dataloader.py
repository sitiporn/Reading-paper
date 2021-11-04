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
from transformers import RobertaTokenizer

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
    

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # check whether example's labels already exist or not 
     
    for e in examples:
        # if they have label in list append sample into the class 

        if len(tokenizer(e.text)['input_ids']) <=7:
            #print(e.text)  
            continue
            
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

def create_supervised_pair(h,labels,debug:bool=False):
    

    """
     h - (batch_size, seq_len, hidden_dim)
     label - (batch_size) 
     create possitive pair 
     eg.  a, b, c, d, e
         
         0 : a, b 
         1 : c
         2 : d,e


     check tags
     2 tensor 
     [a,b,c,d,e] -> masking [[b],[0],[0],[e],[0]]
    idx 0: 
       [0,  0] 
       [12, 24] -> same class 

    skip -> 12, 24  
    
    idx 2:
       [2, 2]
       [13, 25] -> same class
    
    skip -> [12, 24, 13, 25]
    
    """
    # params
    h_i = []
    h_j = []
    skips = []
    T = 0 # the numbers of pairs sample
    
    for idx, label in enumerate(labels): 

        if idx in skips:
            continue

        mask = label == np.array(labels)
        mask[idx] = False 
        
        # check is they have pos pair 
        if np.count_nonzero(mask) >= 1:
            
            idxs_arr = np.arange(len(labels))
            # each h_i and h_j :  (seq_len, hidden_dim)
            
            h_i_tensor = h[idx,:,:]
            h_i_tensor = h_i_tensor[None,:,:]
            h_i_tensor = h_i_tensor.repeat(np.count_nonzero(mask),1,1)
            print("h_i idx :",h_i_tensor.shape)
            print("h_j idx :",h[mask,:,:].shape)
            # (seq_len,hidden_dim) , (#pairs,seq_len, hidden_dim)
            h_i.append(h_i_tensor)
            h_j.append(h[mask,:,:])


            for val in idxs_arr[mask]: 
                skips.append(val)
            # add pair numbers of samples 

            T+= np.count_nonzero(mask)
            
            if debug == True:
                
                #if np.count_nonzero(mask) >= 2:
                print("idx:",idx)
                print("current skips :",idxs_arr[mask])
                print("current labels :",label)

                label_arr = np.array(labels)

                print("pair class :",label_arr[mask])
                print("mask:", mask)
                print("count:",len(mask))
                print("numbers of pairs:",np.count_nonzero(mask))
           

    
    if h_i:
        
        h_i = torch.cat(h_i,dim=0)
        h_j = torch.cat(h_j,dim=0) 
        print("concat list of h_i:",h_i.shape)
        print("concat list of h_j:",h_j.shape)

        return T, h_i, h_j
    else:

        return T, None, None
        
class combine:

    def __init__(self,dataset_name:str=None,few_shot:str=None):
      
        #params
        self.single_dataset = False

        if dataset_name is not None:
            print("dataset :",dataset_name)
            self.datasets = [] 
            self.datasets.append(dataset_name)
            self.single_dataset = True
        else:
            print("Combine datasets !")

            self.datasets = ['ATIS','BANKING77','CLINC150','HWU64','SNIPS']


        if few_shot is not None: 

            self.num_shot = few_shot
        
        else:
            
            self.num_shot = 'train'
        
    def get_examples(self): 

        combine = [] 
        for data in self.datasets:
            train_file_path = f'../../datasets/Few-Shot-Intent-Detection/Datasets/{data}/{self.num_shot}/'
            train_examples = load_intent_examples(train_file_path)
            combine.append(train_examples)

        
        if self.single_dataset == True: 

           flat_combine_list =  combine[0] 

        else: 
            
            flat_combine_list = [item for sublist in combine for item in sublist]
            
            assert len(flat_combine_list) == len(combine[0]) + len(combine[1]) + len(combine[2]) + len(combine[3]) + len(combine[4])


            

        return flat_combine_list 













