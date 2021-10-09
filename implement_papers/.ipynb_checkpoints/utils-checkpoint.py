import  numpy as np 
import  torch 


class IntentExample:
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

class loss(object):

    def __init__(self,u,v,idx_hyper)->None:

        self.N = 3

        # params 
        self.U , self.V   = u, v
        self.temperature = [0.1,0.3,0.5]
        self.lam  = [0.01, 0.03, 0.05] 
        self.idx_hyper = idx_hyper
        self.lml = 1 
        self.l1 = None

    #similarity function 
    def sim(self):

        magnitude_u = np.sqrt(np.sum(np.power(self.U,2)))
        magnitude_v = np.sqrt(np.sum(np.power(self.V,2)))
    
        return (self.U.T @ self.V) / (magnitude_u * magnitude_v) 

    def self_supervised_cl(self): 

         # ToDo
         # dont forget to add index of top_cl variable
        top_param = np.exp(self.sim()/self.temperature[self.idx_hyper])
        print("top param:",top_param)
        bottom_param = np.sum(np.exp(self.sim()/self.temperature[self.idx_hyper]))
        print("bottom_param:",bottom_param)
        self_cl_loss = - (1/self.N) * np.sum(top_param/bottom_param)
        
        return self_cl_loss
    def total_l1():
         
        self.l1 = self.self_supervised_cl() + self.lam[idx_hyper] * self.lml 

        return self.l1


a = np.random.rand(3,2)
b = np.random.rand(3,2)

l1 = loss(a,b,0)


print(l1.self_supervised_cl())
