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
import random

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

def sample(N, examples,train:bool=True):
    labels = {} # unique classes
    

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # check whether example's labels already exist or not 
     
    for e in examples:
       
        if train == True:
            # remove less than five token from sample
            if len(tokenizer(e.text)['input_ids']) <=7:
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
    def __init__(self,labels,text,batch_size,repeated_label:bool=False):
        self.labels = labels
        self.text = text
        self.batch_size = batch_size 
        self.count = 0 
        self.repeated_label = repeated_label

        if self.repeated_label == True:
            self.exist_classes = [] 
            self.label_maps = None 
            self.ids_maps = []
            self.len_data = len(self.labels)
            self.count_batch = 0 
            self.is_left_batch = False
            
            #print("self.len_data ",self.len_data)
            #print("self.len data",self.batch_size)
            
            self.max_count = self.len_data // self.batch_size 

            if self.len_data % self.batch_size !=0:
                self.max_count += 1 
            
            print("the number of maximum of batching :",self.max_count)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        if self.repeated_label == True:
            self.count +=1  
            # it would be clear after call til batch_size  
            self.exist_classes.append(self.labels[idx])
            self.ids_maps.append(idx)


            if self.count_batch == self.max_count - 1:
                self.count_batch = +1 
                #print("self.count_batch :",self.count_batch)
                self.count_batch = 0 

                if self.len_data % self.batch_size !=0: 
                    self.batch_size = self.len_data % self.batch_size
                    self.is_left_batch = True

                #print("change batch size !",self.batch_size)
                #print("LAST batching !")

            if self.count == self.batch_size:

                unique_labels_keys = list(set(self.exist_classes))
                table = [0] * len(unique_labels_keys)
                unique_labels = dict(zip(unique_labels_keys,table))
                
                if self.is_left_batch == True:
                    self.is_left_batch = False
                    self.batch_size = 16  

                else: 
                    self.count_batch += 1
                    #print("count_batch :",self.count_batch)
                
                for class_key in self.exist_classes:
                    unique_labels[class_key] = +1 

                #print("tables of each labels :",unique_labels)

             
                
                for index, key  in enumerate(unique_labels):
                    


                    if unique_labels[key] > 1:

                       print("v>1 :",unique_labels[key])
                       
                       break

                    
                    if index == len(unique_labels.keys()) - 1:
                        
                        
                        while True:
                           
                           pos_idx = random.randint(0,self.len_data-1) 

                           if self.labels[pos_idx] in unique_labels.keys():
                               if self.labels[pos_idx] == self.labels[idx]:
                                   pass

                               else:
                                   #print("old idx :",idx,self.labels[idx])
                                   idx = pos_idx
                                   #print("new idx :",idx,self.labels[idx])
                                   unique_labels[self.labels[idx]] +=1  
                                   #print("statistics tables :",unique_labels)
                                   # replace last token
                                   self.exist_classes[-1] = self.labels[idx]
                                   """
                                   print("==========")
                                   print("len exist_classes :",len(set(self.exist_classes)))
                                   print(self.exist_classes)
                                   """


                                   if len(set(self.exist_classes)) ==  len(self.exist_classes):
                                       print("unique_labels:")
                                       #print(unique_labels)
                                       
                                   
                                   self.count = 0  
                                   self.exist_classes = [] 
                                   self.ids_maps = []
                                
                                   break 
                                      

         
        label = self.labels[idx]
        
        data = self.text[idx]
        
        sample = {"Class": label,"Text": data}


    
        return sample

def create_pair_sample(h_bar,debug:bool=False):
    """
    h ->  a , b, c, a   (batch_size,#embed_size)
    h'->  a', b',c',a'  (batch_size,#embed_size)
    intent_idx -> (0,1,2,0)
    :definition:
    pos_pair : alll the same samples in the batch second forward 
    neg_pair : all the samples in the batch without itself 

    h_i : [a, b, c, a ] for all i batch up to N 
    hi_bar : [a',b',c',a'] for all i batch up to N
    hj_bar : eg. i = 1 : ([a,a,a],[b', c', a']) sum along N for each i 
    hj_bar shape: (batch_size,batch_size-1,embed_size) 
    hi_3d : (batch_size,batch_size-1,embed_size) 

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
    h_i = [] # pos pairs
    h_j = [] # neg pairs
    skips = []

    # proof done 
    # masking correct
    # pair of concat correct



    T = 0 # the numbers of pairs sample
    
    for idx, label in enumerate(labels): 

        if idx in skips:
            continue

        mask = label == np.array(labels)
        # dont include current sample 
        mask[idx] = False 
        
        # check is they have pos pair 
        if np.count_nonzero(mask) >= 1:
            
            idxs_arr = np.arange(len(labels))
            # each h_i and h_j :  (sample, hidden_dim)
            
            # (hidden_dim)
            h_i_tensor = h[idx,:] # got 1 dim 
            # (1,hidden_dim)
            h_i_tensor = h_i_tensor[None,:] # got 2 dim 
            
            # preparing to broadcast up to # repeated labels
            h_i_tensor = h_i_tensor.repeat(np.count_nonzero(mask),1)

            
            #print("h_j idx :",h[mask,:,:].shape)
            # (seq_len,hidden_dim) , (#pairs, hidden_dim)
            if debug:
                if np.count_nonzero(mask) >= 2:
                    print("----")
                    print("masking label debug :",np.array(labels)[mask])
                    print("current labels ",np.array(labels)[idx])
                    print("---")
                
                print(">>>>>>>>>>>>>")
                print("repeat for broadcast :",h_i_tensor.shape)
                print("before append h_i and h_j")
                print("h_i : ",h_i_tensor.shape)
                print("h_j : ",h[mask,:].shape)


            # proof masking are correct they select the same class 
            h_i.append(h_i_tensor)
            h_j.append(h[mask,:])


            for val in idxs_arr[mask]: 
                skips.append(val)
            # add pair numbers of samples 

            T+= np.count_nonzero(mask)
            
            if debug:
                
                print("idx:",idx)
                print("current skips :",idxs_arr[mask])
                print("current labels :",label)

                label_arr = np.array(labels)

                 
                print("pair class :",label_arr[mask])
                print("mask:", mask)
                print("count:",len(mask))
                print("numbers of pairs one label :",np.count_nonzero(mask))
           
    # after ending loop 
    if debug: 

        print("the number of pairs for entire batch:",T) 
        print("pairs see from labels : ",len(labels)-len(set(labels)))
        print("All skippings :",skips)
        print(labels)

        print("---------------------------------------------")
    
    if h_i:
        
       
        h_i = torch.cat(h_i,dim=0)
        h_j = torch.cat(h_j,dim=0) 

        if debug:
            print("concatenate got h_i :",h_i.shape)
            print("concatenate got h_j : ",h_j.shape)
            print("<<<<<<<<<<<<<<<<<<<<<")


        return T, h_i, h_j
    else:

        return T, None, None
        
class combine:

    def __init__(self,dataset_name:str=None,exp_name:str=None,oos_exp:str=None):
      
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

        # eg. 5 shot or 10 shot data 
        if exp_name is not None:

            self.exp_name = exp_name 

        elif oos_exp is not None:

            self.exp_name = 'oos'
            self.oos_exp = oos_exp

            

     
    def get_examples(self): 

        combine = [] 
        
        if self.exp_name != 'oos':
        
             for data in self.datasets:
                train_file_path = f'../../datasets/Few-Shot-Intent-Detection/Datasets/{data}/{self.exp_name}/'
                train_examples = load_intent_examples(train_file_path)
                combine.append(train_examples)
       
        else: 

             for data in self.datasets:
                train_file_path = f'../../datasets/Few-Shot-Intent-Detection/Datasets/{data}/{self.exp_name}/{self.oos_exp}'
                train_examples = load_intent_examples(train_file_path)
                combine.append(train_examples)


        
        if self.single_dataset == True: 

           flat_combine_list =  combine[0] 

        else: 
            
            flat_combine_list = [item for sublist in combine for item in sublist]
            
            assert len(flat_combine_list) == len(combine[0]) + len(combine[1]) + len(combine[2]) + len(combine[3]) + len(combine[4])


            

        return flat_combine_list 













