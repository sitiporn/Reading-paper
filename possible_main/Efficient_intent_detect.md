# Efficient Intent Detection with Dual Sentence Encoders


1. What they did ?

  - leaverage Pretrain and fine-tuning  all entire models 
  - use Conversational pretraining instead of (LM-based)
           L diaglog act prediction
           L utterance generation 
  
  - they do not fine-tune entire models, use fixed sentence encoded by  using ConverT and USE
 
  - stack mulilayer percepton on top of fixed representation, followed by softmax layer for multi-class  

  2.1
        input -> ConverT -> o1
        input -> USE -> o2 

  2.2  concatenate o1 and o2 
  2.3  feed concatenate to same classificaion architecture 
    
  
2. what are good at ?
 
  - they learn i/p/context and relavance follow up(response) 


3) metric 

    - performance 
    - efficiency
    - applicability in few shot scenario  

 
### Dataset 


    - BANKING 77  new single-domain (banking) not in HWU64, and CLINIC150 and fined grained single-domain dataset  
      
      - some intent catagories patially overlap with others which requrire find-grained decision 

      - to capture the corrected intent it's not possible to  rely on semantics  of individual words  



     
