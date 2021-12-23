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
        # among feature last dim 
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
            self.model = RobertaForMaskedLM.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            print(":using RobertaForMaskedLM :")
            print("Vocab size :",self.tokenizer.vocab_size)
            self.start = 0 
            self.end = 2 
            self.mask = 50264
            self.pad = 1  

        else:
            if model_name == 'roberta-base':
                 
                # get tokonizer 
                self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
                #self.config = RobertaConfig()
                 
                # get config from roberta 
                self.config = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True)
               
                if classify:
                   
                   self.config.num_labels = 150 
                   self.model = RobertaForSequenceClassification.from_pretrained("roberta-base", config=self.config)
                   print("Using RobertaForSequenceClassification ...")

                else:

                    self.model = RobertaForMaskedLM(self.config)
                    # roberta tokenizer 
                    # more detail on RobertaConfig()
                    self.start = 0 
                    self.end = 2 
                    self.mask = 50264
                    self.pad = 1  
                        
            if model_name == 'bert-base':
                self.config = BertConfig() 
                self.model = BertForMaskedLM(self.config)
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                # token_id for tokenize purpose 
                # more detail on BertConfig()
                self.start = 101
                self.end = 102
                self.mask = 103
                self.pad = 0
             
            print("Vocab size:",self.config.vocab_size)
  

        self.params = None 
        self.device = device
        self.model = self.model.to(self.device)
        self.model.train()
        self.hidden_state_flag = hidden_state_flag
        self.labels = None  
        self.mask_arr = None 
        self.grads = []
         
    def parameters(self):
        return self.model.parameters()

    def get_model(self):
        return self.model

    def load_model(self,select_model,strict:bool=True):
        
        #ref - https://pytorch.org/tutorials/beginner/saving_loading_models.html
        
        PATH = '../../models/'+ select_model      

        if strict == True:
            self.model.load_state_dict(torch.load(PATH))
            print("Load weight fully match  to model done !")
        else:

            self.model.load_state_dict(torch.load(PATH),strict=False)
            print("Load weight patial done !")

    def freeze_layers(self,freeze_layers_count:int):

        count_layer = 0
        if freeze_layers_count:
        # freeze embeddings of the model
            for param in self.model.roberta.embeddings.parameters():

                param.requires_grad = False 

            if freeze_layers_count != -1:

                for layer in self.model.roberta.encoder.layer[:freeze_layers_count]:

                    count_layer +=1 
                    for param in layer.parameters():
                        param.requires_grad = False
                    
            print("the number of freezing layers",count_layer) 

    def get_grad(self):
        
        self.grads = []

        for param in self.model.parameters():

            self.grads.append(param.grad)

        #self.grads = torch.cat(self.grads)

        return self.grads 

    def eval(self):
        
        return self.model.eval()
           
    def get_label(self,debug:bool=False):

        if debug == True:
            if self.labels is None:
                print("label is None !")

            if self.mask_arr is None:
                print("mask_arr is None !")
                

        return self.labels , self.mask_arr

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
                print("labels id :",self.labels)
                print("class text :",label)

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
            mask_arr = (rand < 0.10) * (inputs['input_ids'] !=self.start) * (inputs['input_ids'] != self.end) * (inputs['input_ids'] != self.pad)
            
            if debug== True:
                print("Masking step:")
                print(mask_arr.shape)
            
            #create selection from mask
            # mask_arr : (batch_size,seq_len)
            inputs['input_ids'][mask_arr] = self.mask
            self.mask_arr = mask_arr  
            #print("Masking checking:")
            #print(self.mask)
            #print(inputs['input_ids'])
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

def contrasive_loss(h,h_bar,hj_bar,h_3d,temp,N,compute_loss:bool=False):

    if compute_loss:

        sim = Similarity(temp=temp)
        pos_sim = torch.exp(sim(h,h_bar))
        neg_sim = torch.exp(sim(h_3d,hj_bar))
        # sum of each neg samples of each *sum over j
        neg_sim = torch.sum(neg_sim,dim=1) 
        
        cost = -1 * torch.log(pos_sim / neg_sim)
 
        """
        hj_bar : eg. i = 1 : ([a,a,a],[b', c', a']) sum along N for each i 
        hj_bar shape: (batch_size,batch_size-1,embed_size) 
        hi_3d : (batch_size,batch_size-1,embed_size) 

        h_neg_bar = [[b',c',a'],[a',c',a'],[a',b',a'],[a',b',c']] 

        """

        return  pos_sim, neg_sim, torch.sum(cost)/N
         
    else:
        sim = Similarity()
        pos_sim = sim(h,h_bar)
        neg_sim = sim(h_3d,hj_bar)
        
        return pos_sim, neg_sim, None
         

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
    if debug:
        print("----------")
        """
        print(": before norm vect :")
        print("h_i :",h_i[:2,:2]) 
        print("h_j :",h_j[:2,:2])
        print("h_n :",h_n[:2,:2])
        print("sim pos :",sim(h_i,h_j)[:2])
        """
    # tested norm without norm sim are the same  
    if callback != None:
       
       h_i = callback(h_i)
       h_j = callback(h_j)
       h_n = callback(h_n)
       """
       if debug:
           print("norm vect")
           print("h_i :",h_i[:2,:2]) 
           print("h_j :",h_j[:2,:2])
           print("h_n :",h_n[:2,:2])
           print("sim norm pos:",sim(h_i,h_j)[:2])
        """ 
      
    # exp(sim(a,b)/ temp)
    pos_sim = torch.exp(sim(h_i,h_j))
    

    """
    :Proof contrasive loss:
    0. masking h_i and h_j pair compute are correct -> done 
    1. with or without norm the sim are the same on positive pairs 
    2.   

    Todo:

    1. check norm with sim 
    2. check bottom factor h_i and h_n: all in the batch  did they braodcast correct
    3. check sum procedure :  over N bottom and over j and over i -> contrasive loss 

    """

    
    # for compute loss 
    bot_sim  = []

    
    for idx in range(h_i.shape[0]):
        
        # broadcast each idx to h_n (all in batch)
        h_i_broad = h_i[idx].repeat(h_n.shape[0],1)
        
        if debug:
            print("h_i before broad :",h_i[idx].shape)
            print("after broad h_i to h_n",h_i_broad.shape)
            print("h_n shape :",h_n.shape)
        
       # sim(hi,hn)/t 
        res = sim(h_i_broad,h_n)



        
        if debug:
            print("sim(h_i,h_n) shape :",res.shape)
            print("neg_sim max_min :",res.max(), res.min())
         
        # sum over batch
        res = torch.sum(torch.exp(res)) 
   
        if debug:
            print("after summing bottom factor :",res.shape)


        # to use with each pair i and j 
        bot_sim.append(res)

    bot_sim = torch.Tensor(bot_sim)

    #neg_sim = neg_sim.reshape(-1,1)


    if debug:
        print("bot_sim :",bot_sim.shape)
        print("pos_sim.shape :",pos_sim.shape)     
    


    loss_s_cl = torch.log(torch.sum(pos_sim/bot_sim))
    loss_s_cl = -loss_s_cl / T   

    if debug:
        print("len(neg) :",len(bot_sim))
        print("loss_s_cl:", loss_s_cl)
        print("")

    

   
    return loss_s_cl 


def get_label_dist(samples, train_examples,train=False):
    
    """
    label_list - list of label texts

    """
    label_map = {samples[i]['task']: i for i in range(len(samples))}
    #label_map = {samples[i].label: i for i in range(len(samples))}
    
    # Hard code -> refactor later 
    # bugs key errors cancel when run from funtuning 
    # label_map['cancel'] = 149 
    #print("label_map:",label_map)
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

def intent_loss(logits,N:int=16,debug:bool=False):

    """
    intent = − 1 (1/N)* sum over log P(Cj|ui) 
    P(Cj|ui)  -  probability of sentence i-th to be j-th class
    shape : (batch_size, config.num_labels) 

    """
    # last dim among the class 
    soft = nn.Softmax(dim=-1)
    # get prob 
    prob = soft(logits)

    if debug:
        print("prob max",prob.max())
        print("prob min",prob.min())

    # vectorize prob with log  
    log_prob = torch.log(prob)
   
    # sum over batch 
    log_prob = torch.sum(log_prob,dim=0) 

    #
    if debug:
        print("log_prob sum over batch:",log_prob.shape)

     

    # sum over class 
    
    loss_intent = -torch.sum(log_prob,dim=-1) / N
    

    return loss_intent 


"""
def intent_klv_loss(label_ids, logits, label_distribution, coeff, device)->Union[ndarray, Tensor]:
   
    i - ith sentence
    j - intent classes
    C - the number of class
    N - the number of intents
    
    p(c|u) = softmax(W h + b) ∈ R
    
    P(Cj|ui)  -  probability of sentence i-th to be j-th class

    inputs_embeds - (batch_size, sequence_length, hidden_size)  
    logits - (batch_size, num_class) 
   
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

"""    
    


