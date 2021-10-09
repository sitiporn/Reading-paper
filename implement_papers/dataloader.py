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

    for e in examples:
        if e.label in labels:
            labels[e.label].append(e.text)
        else:
            labels[e.label] = [e.text]

    sampled_examples = []
    for l in labels:
        random.shuffle(labels[l])
        if l == 'oos':
            examples = labels[l][:N]
        else:
            examples = labels[l][:N]
        sampled_examples.append({'task': l, 'examples': examples})

    return sampled_examples
