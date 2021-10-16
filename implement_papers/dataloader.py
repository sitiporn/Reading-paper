import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import AutoModel, AutoTokenizer
import random
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Type, Union

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
        return self.cos(x, y) / self.temp

class SimCSE(object):
    """
    class for embeddings sentence by using BERT 

    """
    def __init__(self,device):
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.device = device

    def encode(self,sentence:Union[str, List[str]],batch_size : int = 64,max_length:int = 128):
        

        target_device = self.device 
        self.model = self.model.to(target_device)
        
        if isinstance(sentence,str):
            sentence = [sentence]
            single_sentence = True 


        embedding_list = []

        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)

            """
            print(total_batch)
            print(sentence)
            print(type(sentence))
            print(dir(self.tokenizer))
            """
            assert str == type(sentence[0])
            inputs = self.tokenizer(sentence[0],padding=True,truncation=True,max_length=max_length,return_tensors="pt")
            print("Input1:")
            print(inputs)
            # move tensor value to cuda device  
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            print("Input2:",inputs)
            # Encode to get hi the representation of ui  
            outputs = self.model(**inputs, return_dict=True)
            # the shape of last hidden -> (batch_size, sequence_length, hidden_size)
            print("outputs:",outputs.last_hidden_state)

            print("outputs:",outputs.last_hidden_state.shape)
            
                
