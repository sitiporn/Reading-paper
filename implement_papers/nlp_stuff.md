# Natural language processing 


## RNN 
 

  # Embedding
      *ineffective ways
       one hot embedding  #size  -> maximum len <- len(vocab) 
           L  dont capture world 

       *good
           L just prop -> maximum len <-infinity  
           eg. word2vect, fasttext, glove, Elmo, BERT


       how to define target seq is still active research area 

      
       * sparse one-hot embedding 
        [11,4,0,....,2]

        11 -> [0 0 .. 1] at pose 11
