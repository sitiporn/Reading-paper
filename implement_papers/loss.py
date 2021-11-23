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
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from transformers import RobertaConfig, RobertaModel
from torch import linalg as LA
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaForMaskedLM


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp=None):
        super().__init__()

        if temp !=None:
            self.temp = temp
        else: 

            self.temp = 1.0

        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        
        return  self.cos(x, y) / self.temp


class SimCSE(nn.Module):
    """
    class for embeddings sentence by using BERT 

    """
    def __init__(self,device,pretrain:bool = False,hidden_state_flag:bool = True,classify:bool = False,model_name:str='roberta-base'): 
        super(SimCSE,self).__init__()


       # change head to classify 
        self.classify = classify
        
        if pretrain == True: 
            self.model = AutoModel.from_pretrained(model_name)
        else:
            if model_name == 'roberta-base':
                 
                self.config = RobertaConfig()
                self.config.vocab_size = 50265 
                self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
               
                if classify:
                   
                   self.config.num_labels = 150 
                   self.model = RobertaForSequenceClassification(self.config)
                   print("Using RobertaForSequenceClassification ...")

                else:

                    self.model = RobertaForMaskedLM(self.config)
                 
            if model_name == 'bert-base':
                self.config = BertConfig() 
                self.model = BertForMaskedLM(self.config)
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


        print("Vocab size:",self.config.vocab_size)
  

        self.device = device
        self.model = self.model.to(self.device)
        self.model.train()
        self.hidden_state_flag = hidden_state_flag
        self.labels = None  
         
    def parameters(self):
        return self.model.parameters()

    def get_model(self):
        return self.model

    def get_label(self):
        if self.labels is None:
            print("is None")

        return self.labels

    def encode(self,sentence:Union[str, List[str]],label:Union[str,List[str]]=None,label_maps = None,batch_size : int = 64, keepdim: bool = False,max_length:int = 128,debug:bool =False,masking:bool=True,train:bool=True)-> Union[ndarray, Tensor]:
       

        if train == True:
            
            self.model.train()
            #print("Training mode:")

        else:
            self.model.eval()

            print("validation mode mode:")

        single_sentence = False

        if isinstance(sentence,str):
            sentence = [sentence]
            single_sentence = True 
        if debug== True: 
            print(single_sentence)
        embedding_list = []

        #with torch.no_grad():
        total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)

        if debug == True:
            print("Before tokenize",sentence)

        inputs = self.tokenizer(sentence,padding=True,truncation=True,return_tensors="pt")
       
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        

        if self.classify: 
            #print("Classify:")
            if label_maps is not None:
                #print("Label is not none ")
                self.labels = [label_maps[stringtoId] for stringtoId in (label)]
                # convert list to tensor
                self.labels = torch.tensor(self.labels).unsqueeze(0)
                #print(self.labels.shape)
        else:

            self.labels = inputs['input_ids'].detach().clone()
            

        self.labels = self.labels.to(self.device)
        

       
                
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
        outputs = self.model(**inputs,labels=self.labels,output_hidden_states=self.hidden_state_flag)

        # the shape of last hidden -> (batch_size, sequence_length, hidden_size)
        
        hidden_state = outputs.hidden_states 
        hidden_state = hidden_state[12]
       
        # (batch_size, sequence_length, hidden_size)
        embeddings = hidden_state
 

        if debug== True: 

            print("outputs:",len(outputs))
            
            print("hidden states:",embeddings.shape)
             
                   
       
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

def norm_vect(vectors):    
   
    norm = torch.linalg.norm(vectors)

    if norm == 0:
        return vectors


    return vectors/norm




def supervised_contrasive_loss(h_i,h_j,h_n,T,temp,callback=None,debug=False)->Union[ndarray, Tensor]:
    """
    5- shot 
    10 - shot
   
    T - number of pairs from the same classes in batch
   
    each batch
    i - class i
    T = sum_i(N_i-1)
    
    pos_pair - two utterances from the same class
    
    * remark previous work treat the utterance and itself as pos_pair

    neg_pair - two utterances across different class  
   
        loss_stage2   
    """
    
    sim = Similarity(temp)
    
    if callback != None:
       
       h_i = norm_vect(h_i)
       h_j = norm_vect(h_j)
       h_n = norm_vect(h_n)
       
      
    
    pos_sim = torch.exp(sim(h_i,h_j))

    neg_sim  = []
    
    for idx in range(h_i.shape[0]):

        res = sim(h_i[idx].repeat(h_n.shape[0],1,1),h_n)

        #print("neg_sim max_min :",res.max(), res.min())
        
        if debug:
            print("res.shape :",res.shape)
       
        res = torch.sum(torch.exp(res)) 
   
        if debug:
            print("after summing res.shape :",res.shape)

        neg_sim.append(res)

    neg_sim = torch.Tensor(neg_sim)
    neg_sim = neg_sim.reshape(-1,1)


    if debug:
        print("neg_sim.shape :",neg_sim.shape)
        print("pos_sim.shape :",pos_sim.shape)     
    


    loss_s_cl = torch.log(torch.sum(pos_sim/neg_sim))
    loss_s_cl = -loss_s_cl / T   

    if debug:
        print("len(neg) :",len(neg_sim))
        print("loss_s_cl:", loss_s_cl)

    

   
    return loss_s_cl 


def get_label_dist(samples, train_examples,train=False):
    
    """
    label_list - list of label texts

    """
    label_map = {samples[0][i]['task']: i for i in range(len(samples[0]))}
    
    # Hard code -> refactor later 
    label_map['cancel'] = 149 
    

    label_distribution = torch.FloatTensor(len(label_map)).zero_()
    
    for i in range(len(train_examples)):
        
        if train_examples[i].label is None:
            label_id = -1
        else:
            #print(train_examples[i].label)
            label_id = label_map[train_examples[i].label]

        if train:
            label_distribution[label_id] += 1.0 

        if train:


            label_distribution = label_distribution / label_distribution.sum()
     
    return label_distribution, label_map

def intent_classification_loss(label_ids, logits, label_distribution, coeff, device)->Union[ndarray, Tensor]:
   
    """
    i - ith sentence
    j - intent classes
    C - the number of class
    N - the number of intents
    
    p(c|u) = softmax(W h + b) ∈ R
    
    P(Cj|ui)  -  probability of sentence i-th to be j-th class

    inputs_embeds - (batch_size, sequence_length, hidden_size)  
    logits - (batch_size, num_class) 
   
    """
    # label smoothing
     
    #print("label_ids :",label_ids.shape)
    #print("logits:",logits.shape)
    #print("label_distribution",label_distribution.shape)

    label_ids = label_ids.cpu()
    target_distribution = torch.FloatTensor(logits.size()).zero_()
    #print("label_ids.size :",label_ids.size())
    #print("target_distribution.shape: ",target_distribution.shape)    
    
    # loop through batch_size  
    for i in range(label_ids.size(0)):
        # shape - (batch_size, seq_len, vocab_size)
        target_distribution[i, label_ids[i]] = 1.0
    
    # label_distribution - (1,K) - K #class
    # target_distribution - (batch_size, vocab_size)
    # y_ls = (1 - α) * y_hot + α / K
    target_distribution = coeff * label_distribution.unsqueeze(0) + (1.0 - coeff) * target_distribution
    target_distribution = target_distribution.to(device)
    
    #print("target_distribution :",target_distribution.shape)
    # KL-div loss
    prediction = torch.log(torch.softmax(logits, dim=1))
    loss = F.kl_div(prediction, target_distribution, reduction='mean')

    return loss 

    
    


