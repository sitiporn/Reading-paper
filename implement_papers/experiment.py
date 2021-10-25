from transformers import RobertaTokenizer, RobertaForMaskedLM
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

model = RobertaForMaskedLM.from_pretrained('roberta-base')
inputs = tokenizer("The capital of France is <mask>.", return_tensors="pt")
labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]


print(inputs.items())
print(inputs.keys())
print(dir(inputs))
outputs = model(**inputs, labels=labels,output_hidden_states=True)

print(len(outputs.hidden_states))
print(len(outputs.hidden_states[12]))
prediction_logits = outputs.logits
