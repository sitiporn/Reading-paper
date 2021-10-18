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


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) #/ self.temp

class SimCSE(object):
    """
    class for embeddings sentence by using BERT 

    """
    def __init__(self,device):
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.device = device

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

        with torch.no_grad():
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
            print(type(tasks))
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


        
        """
        print("=== masking operation==")
        print(mask) 
        print(hi_bar)
        print(h_bar)
        """

















