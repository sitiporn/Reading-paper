from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray
from typing import List, Dict, Tuple, Type, Union
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.autograd import Variable

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class SimCSE(object):
    """
    class for embeddings sentence by using BERT 

    """
    def __init__(self,device):
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.device = device
    
    def parameters(self):
        return self.model.parameters()

    def encode(self,sentence:Union[str, List[str]],batch_size : int = 64, keepdim: bool = False,max_length:int = 128,debug:bool =False,masking:bool=True)-> Union[ndarray, Tensor]:
        

        target_device = self.device 
        self.model = self.model.to(target_device)
        
        single_sentence = False
        if isinstance(sentence,str):
            sentence = [sentence]
            single_sentence = True 
        if debug== True: 
            print(single_sentence)
        embedding_list = []

       #with torch.no_grad():
        total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)

        """
        print(total_batch)
        print(sentence)
        print(type(sentence))
        print(dir(self.tokenizer))
        """
        #assert str == type(sentence[0])
        if debug == True:
            print("Before tokenize",sentence)

        inputs = self.tokenizer(sentence,padding=True,truncation=True,return_tensors="pt")
       
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        
        if debug== True: 
            print("Input2:",inputs)
        
        if masking == True:
            print("shape of input_ids:")
            print(inputs['input_ids'].shape[1])
            rand = torch.rand(inputs['input_ids'].shape).to(target_device)
            # we random arr less than 0.10
            mask_arr = (rand < 0.10) * (inputs['input_ids'] !=101) * (inputs['input_ids'] != 102)
            
            if debug== True:
                print("Masking step:")
                print(mask_arr)
            
            #create selection from mask
            inputs['input_ids'][mask_arr] = 103
            #selection = torch.flatten((mask_arr).nonzero()).tolist()
            print("after masking")
            print(inputs['input_ids'])



        # Encode to get hi the representation of ui  
        outputs = self.model(**inputs, output_hidden_states=True,return_dict=True)

        # the shape of last hidden -> (batch_size, sequence_length, hidden_size)

        if debug== True: 
            print("outputs:",outputs.last_hidden_state)

            print("outputs:",outputs.last_hidden_state.shape)

        embeddings = outputs.last_hidden_state[:, 0]

        if debug== True: 
            print("embeddings.shpape",embeddings.shape) 
        #print(self.model.eval())
        
        embedding_list.append(embeddings.cpu()) 
            
        embeddings = torch.cat(embedding_list, 0)
        
        if single_sentence and not keepdim:
            embeddings = embeddings[0]
            #print("single sentence")



        return embeddings

def contrasive_loss(h,h_bar,hj_bar,h_3d,temp,N):

        
        sim = Similarity(temp)
        pos_sim = torch.exp(sim(h,h_bar))
        neg_sim = torch.exp(sim(h_3d,hj_bar))


        # sum of each neg samples of each *sum over j
        neg_sim = torch.sum(neg_sim,1) 
        print("find similiarty")
        print(pos_sim.shape)
        print(neg_sim.shape)

        cost = -1 * torch.log(pos_sim / neg_sim)

        print("cost:")
        print(cost.shape)

        return torch.sum(cost)/N 


       
