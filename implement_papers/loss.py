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
from transformers import AutoTokenizer

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
    def __init__(self,device,num_class:int,pretrain:bool = False,hidden_state_flag:bool = True,classify:bool = False,model_name:str='roberta-base'): 
        super(SimCSE,self).__init__()


       # change head to classify 
        self.classify = classify
        
        if pretrain == True: 
            self.model = RobertaForMaskedLM.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            print(": using RobertaForMaskedLM :")
            print("Vocab size :",self.tokenizer.vocab_size)

            self.start = 0 
            self.end = 2 
            self.mask = 50264
            self.pad = 1  

        else:
            if model_name == 'roberta-base':
                print("model_name :", model_name)
               
                if classify:
                   
                   # get config
                   self.config =  AutoConfig.from_pretrained("roberta-base")

                   self.config.num_labels = num_class#150 

                   # get tokeizenier 
                   self.tokenizer = AutoTokenizer.from_pretrained(model_name)

                   # get model 
                   self.model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=self.config.num_labels)

                   print("Download Pretrain from Auto model") 

                   print("model weight :",self.model.roberta.encoder.layer[11].output.dense.weight)
                   

                   self.start = 0 
                   self.end = 2 
                   self.mask = 50264
                   self.pad = 1  


                


                else:

                   print("= :training RobertaForMaskedLM from scratch: =") 
                   print("= : loading config to model : = ")
                     
                   # get tokonizer 
                   self.tokenizer =  RobertaTokenizer.from_pretrained(model_name)
                   
                   # get config from roberta 
                   self.config = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True)

                   self.model = RobertaForMaskedLM(self.config)
                   # roberta tokenizer 
                   # more detail on RobertaConfig()
                   self.start = 0 
                   self.end = 2 
                   self.mask = 50264
                   self.pad = 1  

            elif model_name == "nli":

                pass


            elif model_name == 'bert-base':
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
            print("=== : token usage : ===")
            print("start :",self.start)
            print("end : ",self.end)
            print("mask :",self.mask)
            print("pad : ",self.pad) 


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

    def modify_architecure(self):
        self.model.classifier.out_proj = nn.Linear(in_features=768, out_features=self.config.num_labels, bias=True)
        print("architecture :",self.model.classifier.out_proj)

        self.model = self.model.to(self.device)

    def get_model(self):
        return self.model

    def load_model(self,select_model,strict:bool=True):

        #ref - https://pytorch.org/tutorials/beginner/saving_loading_models.html

        # Todo 
        # 1. load weight first by roberta binary classifcation 
        # 2. then, change the config to same of the number of intents  

        #PATH =  '../../baseline/roberta_nli/pytorch_model.bin'


        PATH = '../../models/'+ select_model      

        print("path:",PATH)

        if strict == True:
            self.model.load_state_dict(torch.load(PATH,map_location='cpu'))

            print("Load weight fully match  to model done !")
        else:

            self.model.load_state_dict(torch.load(PATH,map_location='cpu'),strict=False)
            print("Load weight patial done !")

        self.model.num_labels = self.config.num_labels

        print("show config in labels:",self.model.config.num_labels)
        print("model weight :",self.model.roberta.encoder.layer[11].output.dense.weight)
        print("show model :",self.model.classifier.out_proj) 
        print("show weight fc :",self.model.classifier.out_proj.weight)

    def freeze_layers(self,freeze_layers_count:int):

        count_layer = 0
        if freeze_layers_count:
            # freeze embeddings of the model
            print("freeze embeddings :")
            for param in self.model.roberta.embeddings.parameters():

                param.requires_grad = False 

            if freeze_layers_count != -1:

                print("freeze layers")

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
       
       # if label is None:
       #     print("dont give labels :")
       # if label_maps is None:
       #     print("labels is None :")
       # print("hidden_state_flag :",self.hidden_state_flag)


       # print("masking : ",masking)
       # print("batch_size : ",batch_size) 
       # print("keepdim : ",keepdim)
       # print("max length :",max_length)
       # print("masking :",masking)
       # print("train ",train)



        if train: 
            
            self.model.train()
            #print("Training mode:")

        else:
            self.model.eval()

            #print("validation mode mode:")

        single_sentence = False

        if isinstance(sentence,str):
            sentence = [sentence]
            single_sentence = True 
        if debug== True: 
            print(single_sentence)
        embedding_list = []

        #with torch.no_grad():
        total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)


        inputs = self.tokenizer(sentence,padding=True,truncation=True,return_tensors="pt")
       
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        

        # for classify
        if self.classify: 
            #print("Classify:")
            if label_maps is not None:
                #print("Label is not none ")
                self.labels = [label_maps[stringtoId] for stringtoId in (label)]
                # convert list to tensor
                self.labels = torch.tensor(self.labels).unsqueeze(0)

        else:

            # for language modeling task
            self.labels = inputs['input_ids'].detach().clone()
            

        self.labels = self.labels.to(self.device)
        

       
                
        if debug== True: 
            #print("Input2:",inputs)
            print("inputs.keys()",inputs.keys())
        
        if masking == True:
            

            # input_ids : (batch_size,seq_len)
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

           # print("=== :masking : ===")
           # print(" input_ids :",inputs['input_ids'][:3,:10])
           # print("labels language :",self.labels[:3,:10])


        else:
           # print("=== : without masking : ===")
           # print(" input_ids :",inputs['input_ids'][:3,:10])
           # print("labels language :",self.labels[:3,:10])
           pass
        
            
            
        # Encode to get hi the representation of ui  
        outputs = self.model(**inputs,labels=self.labels,output_hidden_states=self.hidden_state_flag)

        # the shape of last hidden : (batch_size, sequence_length, hidden_size)
            

        hidden_state = outputs.hidden_states 
        
        hidden_state = hidden_state[-1]
       
        # (batch_size, sequence_length, hidden_size)
        embeddings = hidden_state
 

        if debug== True: 

            print("outputs:",len(outputs))
            
             
                   
       
        
        embedding_list.append(embeddings.cpu()) 
        

        embeddings = torch.cat(embedding_list, 0)


        
        if single_sentence and not keepdim:
            embeddings = embeddings[0]
            #print("single sentence")



        return embeddings, outputs

def contrasive_loss(h:Tensor,h_bar:Tensor,temp,N:int,compute_loss:bool=False,debug:bool=False):

    """
        hj_bar : eg. i = 1 : ([a,a,a],[b', c', a']) sum along N for each i 
        hj_bar shape: (batch_size,batch_size-1,embed_size) 
        hi_3d : (batch_size,batch_size-1,embed_size) 

        h_neg_bar = [[b',c',a'],[a',c',a'],[a',b',a'],[a',b',c']] 

    """

    if debug:
        print("temp :",temp)
        print("N :",N)
        print("compute_loss :",compute_loss)
        print("debug :",debug)


    if compute_loss:

        sim = Similarity(temp=temp)
    
        # https://stackoverflow.com/questions/63040954/how-to-extract-and-use-bert-encodings-of-sentences-for-text-similarity-among-sen 
        # use value of CLS token 
        # (batch_size,embed_dim)

        # typically representation 
        h = h[:,0,:] 

        # represenations are masked 
        h_bar = h_bar[:,0,:]

        bot = [] 

        # create pos pairs

        for idx in range(h.shape[0]):

            # copy hi  (batch_size,embed_dim)
            hi_copy =  h[idx].repeat(h.shape[0],1)


            # take expo of similarity between hi and hj_bar : all sample in the batch masked by random 10 percentage

            sum_j = torch.sum(torch.exp(sim(hi_copy,h_bar)))
            bot.append(sum_j)
            
            if debug:
               pass
               # print(" hi_copy :",hi_copy.shape)
               # print(" h_bar  :",h_bar.shape)
               # print("sim(hi_copy,h_bar) :",sim(hi_copy,h_bar).shape)
               # print("== : simimilarity :==",sim(hi_copy,h_bar))
               # print(" max :",sim(hi_copy,h_bar).max())
               # print(" min :",sim(hi_copy,h_bar).min())
               # 
               # print("torch.exp(sim(hi_copy,h_bar)) :",torch.exp(sim(hi_copy,h_bar)))

            

        # convert list to tensors 
        bot = torch.tensor(bot)


        # num_pairs equal to N
        pos_sim = torch.exp(sim(h,h_bar))


        prob = pos_sim / bot
        log_prob = torch.log(prob)
        cost =  -torch.sum(log_prob) / N  


        if debug:
            print("h_i :",h.shape)
            print("h_bar :",h_bar.shape)
            print("pos_sim :",pos_sim.shape)
            print("bot :",bot.shape)
            print("bot : ",bot)
            print("prob :",prob)
            print("log_prob :",log_prob)
            print("cost :",cost)




        return  pos_sim, bot, cost 
    
    #-1 * (torch.sum(cost)/N )) 

         
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




def supervised_contrasive_loss(h_i:Tensor,h_j:Tensor,h_n:Tensor,T:int,temp,idx_yij:List,callback=None,debug=False)->Union[ndarray, Tensor]:
    """
    T - number of pairs from the same classes in batch
    
    pos_pair - two utterances from the same class
    
    * remark previous work treat the utterance and itself as pos_pair

    neg_pair - two utterances across different class  
   
    """
    sim = Similarity(temp)
    


    if callback != None:
       
       h_i = callback(h_i)
       h_j = callback(h_j)
       h_n = callback(h_n)
           
    # exp(sim(a,b)/ temp)
     
    pos_sim = torch.exp(sim(h_i,h_j))
     
    # for collect compute  sum_batch(exp(sim(hi,hn)/t)) 
    bot_sim  = []

    # masking bottom
    # same batch size shape
    #mask = np.arange(h_n.shape[0])

    """
    print("idxes yi=yj :",idx_yij)
    print("len yij",len(idx_yij))
    print("mask :",mask)
    print("# hi:  ",h_i.shape[0])
    """
    for idx in range(h_i.shape[0]):
       

        #mask = mask != idx_yij[idx]

        #h_n_neg = h_n[mask,:]
        # create h_i equal h_n_neg.shape[0] copies

        # select current sample from list pos pairs
        h_i_broad = h_i[idx].repeat(h_n.shape[0],1)
        

        if debug:
            print("h_i before broad :",h_i[idx].shape)
            print("after broad h_i to h_n",h_i_broad.shape)
            print("h_n shape :",h_n.shape)
        

        # sum over batch
        res = torch.sum(torch.exp(sim(h_i_broad,h_n))) 
        
        if debug:
            print("sim(h_i,h_n) shape :",res.shape)
            print("neg_sim max_min :",res.max(), res.min())
         
   
        if debug:
            print("after summing bottom factor :",res.shape)


        # to use with each pair i and j 
        bot_sim.append(res)

    bot_sim = torch.Tensor(bot_sim)

    if debug:
        print("bot_sim :",bot_sim.shape)
        print("pos_sim.shape :",pos_sim.shape)     
    
       

    if debug:
        print("bot sim :",bot_sim.shape)
        print("pos sim :",pos_sim.shape)
    

    loss = torch.log((pos_sim/bot_sim))
    
    
    loss = torch.sum(loss)

    if debug:
        print("after take log: ",loss)
    
    loss = -loss / T   

    
    return loss


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
    # get prob from normalized by softmax 
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


    


