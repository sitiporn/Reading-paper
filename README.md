## Reading-papers

  This will cover the several state of art papaers in NLP field 


## Implementation 

#### Todo 
   - [ ] Pre-train Process
     - [x] Preprocess 
       - [x] combined intent datasets without test sets in the contrastive pre-training
       - [x] remove utterances with less than five tokens
       - [x] create positive and negative pairs 
       
     - [ ] create loss stage 1
       - [x] self supervised contrastive loss
       - [x] mask language modeling loss 

   - [ ] Fine Tunning 
     - [ ] 5 shot 
     - [ ] 10 shot

   - [x] Training on Containner 
   - [x] push trainning history to tensorboard

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
