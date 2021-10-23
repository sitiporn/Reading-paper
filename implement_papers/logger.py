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

    def __init__(self,experiment_name:str=None,model_name:str=None,batch_size:int=64,lr:double=1e-5):
       
        # Todo
        #params
        self.exp_name = experiment_name 
        self.model_name = model_name
        now = datetime.now()
        self.dt_str = now.strftime("%d/%m/%Y_%H:%M") 
        self.hype = {"LR":lr,"B":batch_size}
        self.name = f"{self.model_name}_{self.dt_str}_B={self.hype["B"]}_lr _{self.hype["LR"]}"
        
        # eg. exp_name -> pretrain models
        self.name = f'runs/{sel.exp_name}/{self.name}'

        self.writer = SummaryWriter(self.name)


    def logging(self,running_loss:double,step:int):
        self.writer.add_scalar('Loss/Train',running_loss,step) 


    def flush(self):
        self.writer.flush()
    def close(self):
        self.writer.close()
        

