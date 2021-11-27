import torch
import urllib
import os 
from torch.utils.tensorboard import SummaryWriter 
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from datetime import datetime
from utils import loss
from dataloader import IntentExample
from dataloader import load_intent_examples
from dataloader import sample
from dataloader import InputExample
from loss import SimCSE
from loss import Similarity
from dataloader import SenLoader 
from dataloader import CustomTextDataset
from torch.utils.data import Dataset, DataLoader
from dataloader import create_pair_sample
from loss import contrasive_loss
from transformers import AdamW
from torch.autograd import Variable
from logger import Log
from dataloader import combine
from dataloader import create_supervised_pair
from loss import supervised_contrasive_loss 
from loss import get_label_dist
from loss import intent_classification_loss
from loss import norm_vect 
import yaml 
import ruamel.yaml
import json
import re 


select_model = 'roberta-base_epoch14_B=16_lr=5e-06_25_11_2021_12:07.pth'

embedding1 = SimCSE(device='cuda:1',pretrain=True,model_name='roberta-base')
embedding2 = SimCSE(device='cuda:1',classify=True,model_name='roberta-base')


embedding1.load_model(select_model=select_model,strict=True)
embedding2.load_model(select_model=select_model,strict=False)

print("embedding1.lm_head.decoder.weigths",embedding1.model.lm_head.decoder.weight)

print("embedding2.classifier.out_prof.weigths",embedding2.model.classifier.out_proj.weight)
