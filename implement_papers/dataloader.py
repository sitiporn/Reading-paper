import random

class IntentExample(object):

    def __init__(self, text, label, do_lower_case):
        self.original_text = text
        self.text = text
        self.label = label

        if do_lower_case:
            self.text = self.text.lower()
        
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
