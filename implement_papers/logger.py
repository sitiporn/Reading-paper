import torch

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

'''
file structure
 runs  
  L Pretain_model
       L exp1_model1_B=64_lr_1e-5 
       L exp1_model2_B=32_lr_1e-5
  L experiment_file2
      L exp2_model1_B=64_lr_1e-5 
      L exp2_model2_B=32_lr_1e-5

       


'''


class Log:

    def __init__(self,load_weight:bool,lamb:float,temp:float,experiment_name:str=None,model_name:str=None,batch_size:int=64,lr=1e-5):
       
        # Todo
        #params
        self.exp_name = experiment_name 
        self.model_name = model_name
        now = datetime.now()
        self.dt_str = now.strftime("%d_%m_%Y_%H:%M") 
        self.name = f"Load={load_weight}_{self.model_name}_{self.dt_str}_B={batch_size}_lr _{lr}_lambda={lamb}_temp={temp}"
       
        print("name on tensorboard:",self.name)
        # eg. exp_name -> pretrain models
        
        self.name = f'runs/{self.exp_name}/{self.name}'

        self.writer = SummaryWriter(self.name)


    def logging(self,name:str,scalar_value:float,step:int):
        self.writer.add_scalar(name,scalar_value,step) 


    def flush(self):
        self.writer.flush()
    def close(self):
        self.writer.close()
        

