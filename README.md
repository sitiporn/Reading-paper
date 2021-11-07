## Reading-papers

  This will cover the several state of art papaers in NLP field 


## Implementation 

#### Todo 
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
     - [ ] label smoothing 
     - [ ] Intent Predict probabilty loss
     - [x] Create positive and negative pairs 
     - [x] Create Supervised Contrasive learning loss
     - [ ] Create Intent Classification loss
     - [ ] Training 5 shot and 10 shot

### Bug To fix
   
   1. check simility value dim on feature of text 
        - compute sim value along last dim quite correct(along feature axis)
        - (batch_size,seq_len,embedding_size) -> cosine sim -> (batch_size, seq_len)
        - norm and without norm dont affect to sim value
   2. feed pos and negative pair to proof encoder from Pretrain models
        - exploit encoder could not be able to encode well as pos pair quite high and neg pair quite higher even traing till 30 ephocs
   3. check gradient of each layer of encoder if loss not decrease
   4. check J1 : Supervised contrasive loss 
   5. check J2 : Predict probabilty looss
   
## Using Tensorboard on Remote Server container


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

 3. Usage
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
## Build container on exist image

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



### example of readme 
  - ref - https://www.makeareadme.com/
