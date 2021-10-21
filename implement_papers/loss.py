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
from transformers import BertModel, BertConfig
from transformers import BertTokenizer, BertForMaskedLM

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
    def __init__(self,device,pretrain:bool = False,hidden_state:bool = True):

        if pretrain == True: 
            self.model = AutoModel.from_pretrained('bert-base-uncased')
        else:
            self.config = BertConfig()
            self.model = BertForMaskedLM(self.config)


        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.device = device
        self.model = self.model.to(self.device)
        self.model.train()
        self.hidden_state = hidden_state
    
    def parameters(self):
        return self.model.parameters()

    def encode(self,sentence:Union[str, List[str]],batch_size : int = 64, keepdim: bool = False,max_length:int = 128,debug:bool =False,masking:bool=True)-> Union[ndarray, Tensor]:
        
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
        inputs['labels'] = inputs.input_ids.detach().clone()
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if debug== True: 
            print("Input2:",inputs)
            print("inputs.keys()",inputs.keys())
        
        if masking == True:
            #print("shape of input_ids:")
            #print(inputs['input_ids'].shape[1])
            rand = torch.rand(inputs['input_ids'].shape).to(self.device)
            # we random arr less than 0.10
            mask_arr = (rand < 0.10) * (inputs['input_ids'] !=101) * (inputs['input_ids'] != 102)
            
            if debug== True:
                print("Masking step:")
                print(mask_arr)
            
            #create selection from mask
            inputs['input_ids'][mask_arr] = 103
            #selection = torch.flatten((mask_arr).nonzero()).tolist()
             

        # Encode to get hi the representation of ui  
        outputs = self.model(inputs['input_ids'],attention_mask=inputs['attention_mask'],labels=inputs['labels'],output_hidden_states = self.hidden_state)

        # the shape of last hidden -> (batch_size, sequence_length, hidden_size)
        
        hidden_state = outputs[2]
        hidden_state = hidden_state[12]
       
        # (batch_size, sequence_length, hidden_size)
        embeddings = hidden_state
 

        if debug== True: 

            print("outputs:",len(outputs))
            
            print("hidden states:",embeddings.shape)
             
                   
        # output of the embeddings + one for the output of each layer)
        # (batch_size, sequence_length, hidden_size)
        # initial embedding -> index 0 of turple hidden state
        
        
        print(type(hidden_state))
        print(type(embeddings))
        #print(embeddings)

        if debug== True: 
            print("embeddings.shape",embeddings.shape) 
        #print(self.model.eval())
        
        embedding_list.append(embeddings.cpu()) 
            
        embeddings = torch.cat(embedding_list, 0)
        
        if single_sentence and not keepdim:
            embeddings = embeddings[0]
            #print("single sentence")



        return embeddings, outputs

def contrasive_loss(h,h_bar,hj_bar,h_3d,temp,N):

        
    sim = Similarity(temp)
    pos_sim = torch.exp(sim(h,h_bar))
    neg_sim = torch.exp(sim(h_3d,hj_bar))


    # sum of each neg samples of each *sum over j
    neg_sim = torch.sum(neg_sim,1) 

    cost = -1 * torch.log(pos_sim / neg_sim)

    return torch.sum(cost)/N 

def mask_langauge_loss(M):
    
    """
    P(xm) - Predicted probability of mask token xm over total vocabulary

    M - number of masked tokens in each batch 
    """
    
    #cost = 

    return -1 * torch.sum(cost)/ M

