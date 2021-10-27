from random import shuffle
from transformers import RobertaTokenizer, RobertaForMaskedLM
from torch.utils.data import DataLoader, Dataset
from dataloader import CustomTextDataset, SenLoader, combine, sample
from dataloader import *
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import RobertaConfig


# config
T = 1
N = 100
device = 'cuda:1'
batch_size = 64


# collectors
labels = []
samples = []


# combine all datasets
data = combine()
training_examples = data.get_examples()
sample_tasks = [sample(N,training_examples) for i in range(T)]

# Load sentence
train_loader = SenLoader(sample_tasks)
data = train_loader.get_data()


# Load model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
config = RobertaConfig()
config.vocab_size = 50265 
print("config.vocab_size:",config.vocab_size)
print(dir(config.vocab_size))
model = RobertaForMaskedLM(config) 
#model = model.to(device)
#print(dir(tokenizer))
# change vocab size input of Architecture
'''
model.roberta.embeddings.word_embeddings = nn.Embedding(50265, 768, padding_idx=1)
model.lm_head.decoder = nn.Linear(in_features=768, out_features=50265, bias=True)
'''
# Training 

for i in range(len(data)):
    samples.append(data[i].text_a)
    labels.append(data[i].label)

train_data = CustomTextDataset(labels,samples)
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)

for (idx,batch) in enumerate(train_loader):

     print(len(batch['Class']))
     print(len(batch['Text']))
     sentence = batch['Text']   
     inputs = tokenizer(sentence,padding=True,truncation=True,return_tensors="pt")
     #inputs = {k: v.to(device) for k, v in inputs.items()}
     labels = inputs['input_ids'].detach().clone()
     #labels = labels.to(device)
     print(torch.max(inputs['input_ids']))
     print(torch.min(inputs['input_ids']))
     #print(labels)
     '''
     bugs : index of ranges
      L check embedding_dim
      L input_ids may more than vacab index  
     '''
     outputs = model(**inputs,labels=labels,output_hidden_states=True)
     #print(model.roberta.embeddings.word_embeddings.weight.shape)
     #print(len(tokenizer))
     #print(model.eval())
     
     break




'''
 Is language server in correct environments
 root folder of this project:s
 
'''
