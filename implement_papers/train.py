import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np

from utils import loss
from dataloader import IntentExample
from dataloader import load_intent_examples
from dataloader import sample
from dataloader import InputExample
from loss import SimCSE
from loss import Similarity
from dataloader import SenLoader 
from dataloader import CustomTextDataset
from torch.utils.data import Dataset, DataLoader
from dataloader import create_pair_sample
from loss import contrasive_loss
from transformers import AdamW
from torch.autograd import Variable

# Todo: making a batch that should be able to train 
# 1.) feed all data in batch twice through encoder
# 2.) create lstage1 = self_cl_loss + lamda * mlm_loss

# config
train_file_path = '../../dataset/Few-Shot-Intent-Detection/Datasets/CLINC150/train/'
PATH_to_save = './encoder_net.pth'
N = 100  # number of samples per class (100 full-shot)
T = 1 # number of Trials
temperature = 0.1
batch_size = 32 
labels = []
samples = []
epochs = 15

# 
train_examples = load_intent_examples(train_file_path)
"""
structure of this data  [trials] 
trail -> [dict1,dict2,...,dict#intents]
every dict -> {'task':'lable name','examples':[text1,text2,..,textN]}
"""
sampled_tasks = [sample(N, train_examples) for i in range(T)]

embedding = SimCSE('cuda') 
sim = Similarity(temperature)
train_loader = SenLoader(sampled_tasks)
data  = train_loader.get_data()

for i in range(len(data)):
   samples.append(data[i].text_a)
   labels.append(data[i].label)

for param in embedding.parameters():
    param.requires_grad = True 

optimizer= AdamW(embedding.parameters(), lr=1e-4)
train_data = CustomTextDataset(labels,samples)  
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
"""
 Todo : Programming:  Training 
  1) combined all datasets of intents
  2) drop sentence that less than one sentences 
  3) Trainnig BERT from scatch -> pretrain
  4) create mask language loss  
"""
for epoch in range(epochs):
    
    running_loss = 0.0

    for (idx, batch) in enumerate(train_loader):

        # Print the 'text' data of the batch
        #print(idx, 'data: ', batch, '\n')
        
        
        #print(dir(batch))
        # get hidden representation from ui
        # Todo : proof of emebedding
        # Zero parameter gradients
        optimizer.zero_grad()
        # foward
        h = embedding.encode(batch['Text'],debug=False)
        h_bar = embedding.encode(batch['Text'],debug=False,masking=False)
        hj_bar = create_pair_sample(h_bar,debug=False)    
        hj_bar = [ torch.as_tensor(tensor) for tensor in hj_bar]
        hj_bar = torch.stack(hj_bar)
        
        h_3d = h.unsqueeze(1)
        # print(hj_bar.shape,h_bar3d.shape)
        # print(len(hj_bar),len(hj_bar[0]),len(hj_bar[0][0]))
        # change it to be to taking grad
        
        loss_cl = contrasive_loss(h,h_bar,hj_bar,h_3d,temperature,batch_size) 
        optimizer.step()
        # print statistics
        running_loss += loss_cl.item()

        if idx % 100  == 99: # print every 50 mini-batches
           
            print('[%d, %5d] loss: %.3f' %(epoch+1,idx+1,running_loss/100))
            running_loss = 0.0

        

print('Finished Training')
#torch.save(embedding.state_dict(),PATH_to_save)


"""
sentence = ["get an uber to take me to my brother's house in mineola",'i need to transfer from this account to that one']

embed = embedding.encode(sentence,debug=False)
print(embed)
#print(dir(training_data))

print("Training:",embed)
# Note : same intent sim higher than different intents
# but the diff one not quite well yet
print(sim(embed[0],embed[1]))
print(sim(embed[0],embed[9]))

"""
   

