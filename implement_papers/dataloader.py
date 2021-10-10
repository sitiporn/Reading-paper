from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import random

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

class Classifier:
    def __init__(self,path: str,label_list,args):

        self.label_list = label_list
        self.num_labels = len(self.label_list)

    def convert_examples_to_features(examples, train):
            label_map = {label: i for i, label in enumerate(self.label_list)}
            if_roberta = True if "roberta" in self.config.architectures[0].lower() else False
            
            if train:
                label_distribution = torch.FloatTensor(len(label_map)).zero_()
            else:
                label_distribution = None

            features = []
            for (ex_index, example) in enumerate(examples):
                tokens_a = self.tokenizer.tokenize(example.text_a)

                if len(tokens_a) > self.args.max_seq_length - 2:
                    tokens_a = tokens_a[:(self.args.max_seq_length - 2)]

                tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
                segment_ids = [0] * len(tokens)
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                padding = [0] * (self.args.max_seq_length - len(input_ids))

                input_ids += padding
                input_mask += padding
                segment_ids += padding

                assert len(input_ids) == self.args.max_seq_length
                assert len(input_mask) == self.args.max_seq_length
                assert len(segment_ids) == self.args.max_seq_length

                if example.label is not None:
                    label_id = label_map[example.label]
                else:
                    label_id = -1

                if train:
                    label_distribution[label_id] += 1.0
                    
                features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_id=label_id))
                
            if train:
                label_distribution = label_distribution / label_distribution.sum()
                return features, label_distribution
            else:
                return features
