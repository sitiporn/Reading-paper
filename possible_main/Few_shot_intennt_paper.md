# Few-shot Intent Detection via Contrasive Pre-trainning and Fine-Tuning 


   * Methodology 

    CPFT methodology
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
        
