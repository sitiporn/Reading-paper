## Reading-papers

  This will cover reading and implementing the several state of art papaers in NLP field 

### terms 
    - task 0: pretrain of encoders for language understanding 
    - task 1: pretrain of encoders for understanding utterances and contrastive utteraces tasks without labels 
    - task 2: fine-tunning tasks (eg. few shot) train on KL-div predict probabilty and contrastive utteraces with labels 

### Urgent Plan
   - Pretrain(self-supervised)
        - [ ]  1.training Pretrain validate each epochs
          - from scartch
            - roberta-base_epoch14_B=16_lr=5e-06_24_11_2021_11:34.pth  
              - Suck on language understanding task cannot predict mask token
              - on Similarity -> pending
            -   - Load=False_roberta-base_epoch14_B=16_lr=5e-06_08_12_2021_07:21.pth
          - existed weight
            - roberta-base_epoch14_B=16_lr=5e-06_25_11_2021_04:23.pth
            - roberta-base_epoch14_B=16_lr=5e-06_25_11_2021_12:07.pth (recently) 
              - on language understanding  (of last epochs validation 31 %) ,( after training session use validation set 41 %)
              - on similarity 
          
     
            
        - [ ]  2.test RobertaCasualLm on test set on both validation and test set to on similarity, predict <mask> token
        - [ ]  3.Load Pretrain and train on dataset of this papers validate each epochs
        - [ ]  4.test from (3.) RobertaCasualLm on test set on both validation and test set to on similarity, predict <mask> token
        - [ ]  5.compare the validation acc growing both training option and choose the best one.
        - [ ]  6.if things go well, the sim of positive pairs should be 0.8 - 0.9 and negative pairs should be negative  and also predict token_ids quite high precision -> to proof pretrain models is not sucked -> go to fine tune state 
  
   - Fine Tune
     - roberta-base_B=16_lr=5e-06_27_11_2021_07:20.pth (config.yaml)
     - roberta-base_B=16_lr=0.001_27_11_2021_17:06.pth
     - roberta-base_B=32_lr=5e-06_01_12_2021_11:48.pth
   
   - Furthermore work
     - [ ] downloading baseline model to proof that dataset is able to discriminate  on CLINIC150, BANKINK77, HWU64
     - [ ] fill-in validation acc on Pretrain dataset and fine-tunning dataset on CLINIC150, BANKINK77, HWU64
     - [ ] and test acc of every datasets

### Assumption on theory 
    - on tasks 1 whether we should use weights from task 0 or not is depend on utteraces understanding on tasks 1 performance if they cannot predict hidden masks we should use weight from task 0 if performance still ok no need to do so 
    - unfreeze all layers to train on tasks 1 as we have a lot of samples  
    - on task 2 train only classifier head as we have few samples  

### Discussion 

## Implementation 


### Experiment instruction
    Dataset
     - Pretrain Datasets
       1. combine all dataset 80,782 utterances
       2. remove less than five tokens
       3. conduct self-supervised pretrainning on collected utterances without using labels.
       
     - Evaluation Datasets
       
       1. compare CLINC150 contains 23,700 utteraces 10 domains
       2. BANKING77 13,083 single banking domain 77 intents
       3. HWU64 25,716 utterances 64 intents spaning 21 domains follow setup (https://arxiv.org/pdf/2010.08684.pdf) small portion of trainning sets separated as 
       validation set and test set is unchanged repeat few-shot learning model 5 times and report average accuracy.
      
### Sub Experiment setup (Example-Driven Intent prediction with Observers)
 
   (1) using the full training set and 
   (2) using 10 examples per intent or approximately 10% of the training data. 
       
   - both setting evaluate validation set at the end of each epochs 
   - few-shot experiment are more sensitive to initialization and hyperparameters
   
    ref - https://arxiv.org/pdf/2010.08684.pdf
    
### Training instruction
     1. Pretrain combine intents without test set  in the contrastive pre-training stage for 15 epochs bach = 64, τ to 0.1, and λ to 1.0
     2. Fine tune 5-shot, 10-shot, batch = 16, and do hyperparameters seach τ and λ′ 30 epochs and apply label smoothing
     
       
### Experiment settings
    
     -  under 5-shot (5 training examples per intents) 
     -  10-shot settings (10 training examples per intents)

     -  batch size to 16, and
     -  τ ∈ {0.1, 0.3, 0.5}
     -  λ′ ∈ {0.01, 0.03, 0.05};  
     -  fine-tuning takes 30 epochs
     -  label smoothing to the intent classification loss Zhang et al. (2020a).

    
### Bug To fix

   check gradient of each layer of encoder if loss not decrease
   check J1 : Supervised contrasive loss 
   check J2 : Predict probabilty looss

### Bug report 
   - [x] use [CLS] to represent the whole seq and calculate sim without normalize vectors 
      - the sim same sen around 0.9 and diff 0.7
      - the sentence from the same class and different class encode the sim value 0.9 and 0.7 which is not much different 
      -  feed pos and negative pair to proof encoder from Pretrain models
           - exploit encoder could not be able to encode well as pos pair quite high and neg pair quite higher even traing till 30 
           - ephocs

### Bug found
     - simcse class define token_ids wrong which affects on pretrain and fine-tunning model on roberta-base 

  references
  - https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn
  - https://dev.to/iggredible/debugging-in-vim-with-vimspector-4n0m

## Result
Model | %Test Acc  | %Train Acc| %Validation | Hyperparameter usage  | 
--- | --- | --- | --- | --- |
Roberta | %19|   |    | temp:0.3 , lamda:0.05  | 
Roberta | %8 |   |    | temp:0.5 , lamda:0.05 |
Roberta | %12 |  |    | temp:0.5 , lamda:1.0 |

- Further result
  - https://docs.google.com/spreadsheets/d/1eGla8CvHOVMP_I3NML3YlgwF-GE9hDUb-k4FgmJeDrE/edit#gid=975089668                                                                          
 - Exmaple of validate pretrain on language understanding tasks
  ![image](https://user-images.githubusercontent.com/31414731/143609508-d28b5cd1-46f2-4248-b744-cdf5ab799faa.png)                           

## Usage

### Run Background                                                 
```bash
nohup python3 pretrain_test.py > output.log &                                                                     ```
```
ref - https://janakiev.com/blog/python-background/                                                                            

### Using Tensorboard on Remote Server container


 1. map the remote port to a local port run on local machine

```bash
ssh -L 6006:localhost:6006 containner_name
```
 2. runs in the container where your runs file is 

```bash
pip3 install tensorboard
```
```bash
tensorboard dev upload --logdir runs
```

 3. Usage tensorboard
    ```python
     from torch.utils.tensorboard import SummaryWriter
     
     #Writer will output to ./runs/ directory by default
     #can add comment eg. SummaryWriter(comment="LR_0.1_BATCH_16")
     writer = SummaryWriter()  
     # tag of graph    
     # y-axis value <- running_loss
     # x-axis value <- running_time

     writer.add_scalar('Loss/train', running_loss,running_time)

     writer.close()
 
     ```
### Build container on exist image

1. build container

```bash
docker run --name containner_name -d -v /home/corse/st121xxx/thesis:/root/thesis  -it --shm-size=2G -p 2008:22 -p 2021:8888 --gpus all docker_img_name 
```

2. Go from remote server to container directly

```bash
docker exec -it [container name] bash
```

3. change authorize_keys

4. testing foward port 

```bash
ssh root@localhost -p [port]

```

#### What I have done
   - [x] Pre-train Process
     - [x] Preprocess 
       - [x] combined intent datasets without test sets in the contrastive pre-training
       - [x] remove utterances with less than five tokens
       - [x] create positive and negative pairs 
    
      - [x]  Traning bert base on Container
      - [x]  Training roberta base on Container
      - [x]  push trainning history to tensorboard

       
     - [x] create loss stage 1
       - [x] self supervised contrastive loss
       - [x] mask language modeling loss 

   - [ ] Fine Tunning 
     - [x] Convert example to feature  to get label distribution and train feature
     - [x] label smoothing 
     - [x] Intent Predict probabilty loss
     - [x] Create positive and negative pairs 
     - [x] Create Supervised Contrasive learning loss
     - [x] Create Intent Classification loss
     - [ ] Training 5 shot and 10 shot
   

## Tenative plan
- [ ]  Literature Review  
- [x]  draft implement
- [ ]  Prof of concept 
- [ ]  Submission of Proposal Draft (30 Dec)
- [ ]  Proposal Defense (15 june)
- [x]  Data collection 
- [ ]  Data Analysis
- [ ]  progressive Defense ( 20 March)
- [ ]  Analysis and Review
- [ ]  Thesis Draft Submssion 
- [ ]  Final Defense ( 1 May)
- [ ]  Publication (NeurIPS MAY 21) ?

## Time Left (Final Defense 1 April)
 - 121 days 
 - 4 month and 0 days


### example of readme 
  - ref - https://www.makeareadme.com/
