import  numpy as np 
import  torch 



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



