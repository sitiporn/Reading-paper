# Few-shot Intent Detection via Contrasive Pre-trainning and Fine-Tuning 


- CPFT methodology
   - C intents
   - they sets balance K-shot for each intents 
   - each intents K samples 
   - total samples C*K samples



 3.1 Self-supvised Pretrainning 

       i-th - user utterance through an encoder model 
       h - feture representation  
       hi = BERT(ui)
        
       - they use self-supvised to learn and understanding sentence level of utterance 
        
                
       - they use supervised contrasive learnnin to discriminate 
       semantically similar utterance 

       
       eq1 notation detail
           T - control penalty to negative samples
           sim(hi,h_pi) - the cosine similarity between two i/p vectors

       hi_p - represents the representation of sentence ui_p 
       ui_p - where ui_p is from same sentence ui  but few 10% tokens are randomly masked 


       - They dynamically maksked tokens across different position during trainning each epochs  
           L it benefits utterances understanding 

       - (ui,ui_p) i/p -> single encoder 
       
       eq2 P(Xm) - Predictied probability of a maksked tokens Xm over the total vocabulary         

       M - the total number of masked tokens in each batch
        

       Lstage1 = Luns_cl + lamda * Lmlm

 3.2 Supervised Fine-tunning 
          
       positive pair  = two samples from same intents or classes 
       negative pair  = two samples from across different intents classes 

       * in previous work they only use positive pair which make representation different according to dropout in BERT  (possitive pair -> single encoder)        

       guess -> they use both positive and negative pair  to single encoder 

       eq3: 
         T = from the same class in batch

       eq4:

         predicted probability of i-th sentence  to be in class j 

       - they jointly train two losses together at each batch
           :Lstage2 = Ls_cl + lamda * Lintent   
            
           lamda = weight hyperparamter 

4.2 Model Training and Baseline 
     
     Pretrainning stage 

       - 15 epochs
       - batch = 64 

     Fine-tunning 
       - 5 shot -> 5 examples per intents
       - set batch size to 16 
       - do parameters search T and lamda 
       - apply label smoothing to intent classification loss 

      


