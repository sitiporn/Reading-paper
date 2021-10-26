from random import shuffle
from transformers import RobertaTokenizer, RobertaForMaskedLM
from torch.utils.data import DataLoader, Dataset
from dataloader import CustomTextDataset, SenLoader, combine, sample
from dataloader import *
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import RobertaConfig


T = 1
N = 100
labels = []
samples = []
# combine all datasets
data = combine()
batch_size = 64
training_examples = data.get_examples()
#print(type(training_examples))
#print(len(training_examples))
#print(type(training_examples[0]))

sample_tasks = [sample(N,training_examples) for i in range(T)]

train_loader = SenLoader(sample_tasks)
data = train_loader.get_data()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
config = RobertaConfig()
model = RobertaForMaskedLM(config) 

for i in range(len(data)):
    samples.append(data[i].text_a)
    labels.append(data[i].label)

train_data = CustomTextDataset(labels,samples)
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)

for (idx,batch) in enumerate(train_loader):

     print(len(batch['Class']))
     print(len(batch['Text']))
    
     #outputs = model(**inputs,labels=labels,output_hidden_states=True)




'''
print(type(data))
print("len(data):",len(data))
print(type(data[0]))

for i in range(len(data)):
     print(data[i].text_a)
     print(data[i].label)

     if i == 5:
         break

print(type(sample_tasks))
 
print(len(sample_tasks[0][0]))
print(len(sample_tasks[0][1]))
print(type(sample_tasks[0][0]))
print(
#print(type(sample_tasks[0]))


device = 'cuda:1' 





model = RobertaForMaskedLM.from_pretrained('roberta-base')
inputs = tokenizer("The capital of France is Paris", return_tensors="pt")

model = model.to(device)


labels = inputs.input_ids.detach().clone()
inputs = {k: v.to(device) for k, v in inputs.items()}
labels = labels.to(device)
    


print(inputs.items())
print(inputs.keys())
print(dir(inputs))
outputs = model(**inputs, labels=labels,output_hidden_states=True)

print(len(outputs.hidden_states))
print(len(outputs.hidden_states[12]))
prediction_logits = outputs.logits

'''




'''
 Is language server in correct environments
 root folder of this project:s
 
'''
