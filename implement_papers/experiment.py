from transformers import RobertaTokenizer, RobertaForMaskedLM

T = 1
# combine all datasets
data = combine()

training_examples = data.get_examples()

sample_tasks = [sameple(N,training_examples) for i in range(T)]

device = 'cuda:2' 

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')




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
