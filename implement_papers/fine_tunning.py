import torch
import urllib
import os 
from torch.utils.tensorboard import SummaryWriter 
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from datetime import datetime
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
from logger import Log
from dataloader import combine
from dataloader import create_supervised_pair
from loss import supervised_contrasive_loss 
from loss import get_label_dist
from loss import norm_vect 
from loss import intent_loss
from read_config import read_file_config 
#from contrasive_test import compute_sim
import pprint 
import csv 



# collector

train_labels = []
train_samples = []

test_labels = []
test_samples = []

# config

debug = False 
exp_name = 'train_5'
# J1,J2,JT

comment = 'JT'+ exp_name

running_time = 0.0 
freeze_num = [12] #[4,5]  
temp =  [0.1]   # [0.1, 0.3, 0.5]
lamda = [0.5]    #[0.01,0.03,0.05]

path_test = 'config/test.yaml' 
path_finetuning = 'config/config3.yaml'

correct = 0
total = 0 

load_weight= True # existed of task 0 
select_model = 'Load=False_roberta-base_B=64_lr=5e-06_03_03_2022_14:13.pth'
#'roberta-base_epoch14_B=16_lr=5e-06_25_11_2021_12:07.pth'
#'Load=False_roberta-base_epoch14_B=16_lr=5e-06_05_01_2022_12:43.pth' 
#'roberta-base_epoch14_B=16_lr=5e-06_25_11_2021_12:07.pth'


yaml_data = read_file_config(path=path_finetuning)
yaml_test = read_file_config(path=path_test) 

pp = pprint.PrettyPrinter(indent=4)

print(": read yaml fine tunning :")
pp.pprint(yaml_data)

print(": read yaml testing :")
pp.pprint(yaml_test)


# collector variables

# get dataset  
data = combine('CLINC150',exp_name=exp_name) 
test_set = combine('CLINC150','test')


# load all datasets 
train_examples = data.get_examples()
test_examples = test_set.get_examples()

print("len of datasets :",len(train_examples))
print("len of test set :",len(test_examples))

#sample datasets
sampled_tasks = sample(yaml_data["training_params"]["N"], examples=train_examples,train=False) 
sampled_test = sample(yaml_test["testing_params"]["N"],examples=test_examples,train=False)

print("the numbers of intents",len(sampled_tasks))

label_distribution, label_map = get_label_dist(sampled_tasks,train_examples,train=False)

train_loader = SenLoader(sampled_tasks)
test_loader = SenLoader(sampled_test)

"""
print(": label map :")
print(label_map)
"""

"""
#data = train_loader.get_data()

#print(embedding.eval())
"""


# inputs of custom text dataset  
for i in range(len(train_examples)):
   train_samples.append(train_examples[i].text)
   train_labels.append(train_examples[i].label)

for i in range(len(test_examples)):
   test_samples.append(test_examples[i].text)
   test_labels.append(test_examples[i].label)

#get unique class

num_classes = len(np.unique(np.array(train_labels)))
print("the numbers of classes :",num_classes)

# inputs of dataloader 
train_data = CustomTextDataset(train_labels,train_samples,batch_size=yaml_data["training_params"]["batch_size"],repeated_label=True)  

test_data = CustomTextDataset(test_labels,test_samples,batch_size=yaml_test["testing_params"]["batch_size"])  

# Dataloader 

train_loader = DataLoader(train_data,batch_size=yaml_data["training_params"]["batch_size"],shuffle=True)
print("Train Loader Done !")
test_loader =  DataLoader(test_data,batch_size=yaml_test["testing_params"]["batch_size"],shuffle=False,num_workers=2)
print("Test Loader Done !")

# collect sampleing all pos pair after training
table = {str(k):0 for k in range(num_classes)}
#file = open("pos_pair.csv","w")   




for freeze_i in freeze_num:  
    for lam in lamda: 
        for tmp in temp: 


            now = datetime.now()  # get time 
            dt_str = now.strftime("%d_%m_%Y_%H:%M")


            # create dummy model 
            
            embedding = SimCSE(device=yaml_data["training_params"]["device"],classify=yaml_data["model_params"]["classify"],model_name=yaml_data["model_params"]["model"]) 

            embedding.load_model(select_model=select_model,strict=False)

            embedding.freeze_layers(freeze_layers_count=freeze_i)
            print("Freeze Backboned layers",freeze_i)
            print("lamda :",lam) #yaml_data["training_params"]["lamda"])
            print("temperature :",tmp) #yaml_data["training_params"]["temp"])
             
                    
            # Tensorboard
            logger = Log(load_weight=load_weight,num_freeze=freeze_i,lamb=lam,temp=tmp,experiment_name=yaml_data["model_params"]["exp_name"],model_name=yaml_data["model_params"]["model"],batch_size=yaml_data["training_params"]["batch_size"],lr=yaml_data["training_params"]["lr"],comment=comment)




            # create optimizer
            optimizer= AdamW(embedding.parameters(), lr=yaml_data["training_params"]["lr"])

            total = 0
            skip_time = 0 


            for epoch in range(yaml_data["training_params"]["n_epochs"]):

                running_loss = 0.0
                running_loss_s_cl = 0.0
                running_loss_intent = 0.0

                # collect pair sampling summary for each epoch
               
                 

                for (idx, batch) in enumerate(train_loader): 
                

                    optimizer.zero_grad()

                    total +=1


                    print("batch_id :",idx)
                    label_idx = [label_map[stringtoId] for stringtoId in (batch["Class"])]
                    print("batch class :",label_idx)
                    
                    pair_table = {str(k):0 for k in range(num_classes)}



                    for v in label_idx:
                        pair_table[str(v)] +=1 
                        
                        if pair_table[str(v)] >=2:

                            table[str(v)] +=1
                    
                    #print("pos_table [{},{}] :".format(epoch,idx))
                    
                    
                    # (batch_size, seq_len, hidhen_dim) 

                    if len(set(batch['Class'])) == len(batch['Class']):
                        
                        print("no positive pairs !")
                        print("len batch class o/p :",len(set(batch['Class'])))
                        print(batch['Class'])
                        print("=====================")

                    # change the number of class 
                    embedding.modify_architecure()


                    h, outputs = embedding.encode(batch['Text'],batch['Class'],label_maps=label_map,masking=False)
                    
                    # https://stackoverflow.com/questions/63040954/how-to-extract-and-use-bert-encodings-of-sentences-for-text-similarity-among-sen 
                    # use value of CLS token 
                    h = h[:,0,:]
                    

                    T, h_i, h_j, idx_yij = create_supervised_pair(h,batch['Class'],debug=False)
                    # (batch_size, seq_len, vocab_size) 
                    logits = outputs.logits
                     
                                    
                 
                    if h_i is None:
                        print("skip this batch")
                        skip_time +=1
                        continue
               
                    # Todo: debug supervised contrasive loss 
                    loss_s_cl = supervised_contrasive_loss(h_i, h_j, h, T,temp=tmp,idx_yij=idx_yij,debug=False) 

                    #loss_s_cl = compute_sim(h,labels=batch['Class'],temp=yaml_data["training_params"]["temp"])

                    #label_ids, _  = embedding.get_label()
                      
                   
                    #loss_intent = intent_classification_loss(label_ids, logits, label_distribution, coeff=yaml_data["training_params"]["smoothness"], device=yaml_data["training_params"]["device"])
                    """
                    Todo: classifier

                    add on : 
                    - make prediction:
                      - p = Softmax(Mq) : the Mq -> sort of projection of Query to every class  

                    1. p = Softmax(W @ f(x) + b) w_init = [mu1,mu2,mu3] and b = 0 
                    2. Entropy regularization : average of H(p), for all quries 
                    3. Cosine similarity + softmax  just norm method |w| and |q| 
                    """

                    #loss_intent  = intent_loss(outputs.logits,N=yaml_data["training_params"]["batch_size"])
                    loss_intent = outputs.loss 

                    # JT = J1 + J2 

                    loss_stage2 = loss_s_cl + (lam * loss_intent)

                     #(yaml_data["training_params"]["lamda"] * loss_intent)
                    
                    
                    loss_stage2.backward()
                    #print(embedding.get_grad())
                    optimizer.step()
                    # collect for visualize 
                    running_loss += loss_stage2.item()
                    running_loss_intent += ( loss_intent.item())
                    running_loss_s_cl += loss_s_cl.item()
                    
                    if idx % yaml_data["training_params"]["running_times"] ==  yaml_data["training_params"]["running_times"]-1: # print every 50 mini-batches
                        running_time += 1
                        logger.logging('Loss/Train',running_loss,running_time)
                        print('[%d, %5d] loss_total: %.3f loss_supervised_contrasive:  %.3f loss_intent :%.3f ' %(epoch+1,idx+1,running_loss/yaml_data["training_params"]["running_times"] ,running_loss_s_cl/yaml_data["training_params"]["running_times"] ,running_loss_intent/yaml_data["training_params"]["running_times"]))
                        
                        print("skip_time:",skip_time)
                        print("total :",total)

                        #print('[%d, %5d] loss_total: %.3f' %(epoch+1,idx+1,running_loss/running_times))
                        running_loss = 0.0
                        logger.close()
                        model = embedding.get_model()   
                        


                
#            del logger    
            
#            print("delete logger for one combination")
#            print('Finished Training')
#
#            with torch.no_grad():
#                for (idx, batch) in enumerate(test_loader): 
#                    
#                        _, outputs = embedding.encode(batch['Text'],batch['Class'],label_maps=label_map,masking=False,train=False)
#
#                        logits = outputs.logits
#                        logits_soft = torch.softmax(logits,dim=-1)
#
#                        _, predicted = torch.max(logits_soft,dim=-1)
#
#                        if label_map is not None: 
#                            
#                            labels = [label_map[stringtoId] for stringtoId in (batch['Class'])]
#                            labels = torch.tensor(labels).unsqueeze(0)               
#                            total += labels.size(0)
#                            labels = labels.to(yaml_data["training_params"]["device"])
#                            correct += (predicted == labels).sum().item()
#                        
#                        if debug:
#                            print(">>>>>>>>>>")
#                            print("labels: ",labels)
#                            print("class :",batch['Class'])
#                            print("<<<<<<<<<<")
#
#                            print("Predicted:",predicted)
#                        
#
#                        print("Correct :",correct)
#                        print("Total :",total)
#
#                            #break
#
#            print("%Acc :",(correct/total)*100)
#
#            PATH_to_save = f'../../models/{comment}_Load={load_weight}_{yaml_data["model_params"]["model"]}_freeze={freeze_num}_B={yaml_data["training_params"]["batch_size"]}_lr={yaml_data["training_params"]["lr"]}_{dt_str}.pth'
#
#            print(PATH_to_save)
#            print("correct :",correct)
#            torch.save(model.state_dict(),PATH_to_save)
#            print("Saving Done !")
#
#
