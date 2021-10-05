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

                


