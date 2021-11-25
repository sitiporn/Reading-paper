from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch 
import torch.nn as nn 


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')

m = nn.Softmax(dim=-1)

texts = ["I love <mask>.","<mask> hate you.","I like <mask>.","I will <mask> some jobs"]
labels = ["I love you.","I hate you.","I like you.","I will apply some jobs"]

inputs = tokenizer(texts,padding=True,truncation=True,return_tensors="pt")
labels = tokenizer(labels,padding=True,truncation=True,return_tensors="pt")["input_ids"]

mask = inputs.input_ids == 50264

print("mask:",mask)

outputs = model(**inputs, labels=labels)

loss = outputs.loss
logits = outputs.logits

prediction = m(logits) 
print(prediction.shape)

prediction = torch.max(prediction,dim=-1)[1]

print(prediction.shape)

print("inputs['input_ids'] :",inputs['input_ids'])
print("labels :",labels)
print("prediction :",prediction)
print("prediction",prediction[mask])
print("labels",labels[mask])
print("Compare :")
print(prediction[mask] == labels[mask])
