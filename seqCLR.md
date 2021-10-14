# Contrastive Representation Learning for Electroencephalogram Classification


   what is the key idea of this paper ?
      
      - On pretext tasks they adopted SimCLR (self-supervised contrasive leanring) 
       

    :Methodology:
          
          1) They did data Augmentation in different way before past through 
             from the same signal 


          2) They pass it through the Encoders 
          3) They use those Encoders to use it on downstream taksk eg. classification task 

          4) They adopted Encoders to use then freeze weight of this part during fine-tunning 
    

    : the best combination: 
         
          Convolutiontional Encoders with projectors

    : The most important Architecture: 
        
           EEG channel -> random data Augmentation pair -> channels Enconers -> projectors - > maiximize Argreements

    

    :objective function:
          
          On Pretext tasks
             L they should be able to maiximize the similirity value and should be able to know that this from the same examples
   

    # Datat Augmentation part  
     

        there are many kinds of Augmentation data 
           eg.  Time-shift , band stop, masking EEG signals, Scaling Ampitude, 
           DC shift, re-combination signal 



          - recombination signal  
               
               bring singnal eg.channel a - channel b 
               what they did just recombination and create new channels 
                  eg. ch_a - ch_b = ch_c
                  
    

    # Alabation study 
       
        - Removing the process of scaling and masking signal effect is effect on accuracy the most 
