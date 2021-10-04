# 15 min each paper

#1. Self-Supervised Learning of Pretext-Invariant Representations

 - pretexts -> covariant with img transformations <= may be problem
 - semantic represenation -> should invarinat under such transform
                                L > develop PIRL <- method
 - why it invariant are important in such transformation ?
        guess -> objective to know they come from the same img
                                                    L > help them learn good representation

 - what is pretext tasks ?
       - it's the self-supervised tasks is solve to learn useful representation 
       - useful -> easily to adapt to other tasks     L > learned by learning objective function of     objective tasks

  1) what are the tasks problems or what are the next challenges ?
      - richer sets of transformation 
      - combianation PIRL with clustering-based approaches 
              L their next assumption -> better image representation 
  2) what problem that they solve ?
     -  It -> representation of It that are predicted by convNet 
                    L -> covary -> not contain much semantic representation 
                                                                  
  3) How they solve ?
            L they use PIRL - invariant to transfrom t and retain semantic information 
  4) the usage of these papers
          image recognition 
             L image classification
             L object detection
* the representation are covary with transformation may loose 
semantics information

#2. Barlow Twins: Self-Supervised Learning via Redundancy Reduction

  1) what are the tasks problems or what are the next challenges ?

  2) what problem that they solve ?
       - trivial constant collapse mode 
       - the robustness of trainning batch size 
  3) How they solve ?
      -use cross correlation matrix between o/p of two identical Netwk 
    feed two identical network feed with distorted a sample
      - objective  
          L make it close to identity matrix 
  
  4) the usage of these papers 
     - Is on par state of art for imageNet with a linear classifer
     what is linear classifier ?
     - transfer tasks of classification and object detection

    *objective fucntion -> measures the cross correlation matrix between embedding of two identical networks
    
    * try to make this matrix to identity 
    * loss between (C,I) 
         C -> cross correltation matrix 
         I -> Target cross-corr.

#3. Audio-Visual Instance Discrimination with Cross-Modal Agreement
 
  audio-video -> "instance-based"  
    -learn to align video and audio instances

  AVID -> didnt optimize for visual similarities
  Using CMA -> calibrate to calibrate AVID  
                 L videos in group have to similiar in vid sim  and Audo sim space
                 L which directly optimize visual representation 

                 

  1) what are the tasks problems or what are the next challenges ?
  limitation of AVID 
  2) what problem that they solve ?
       recent work, contrasive learning define positive and negative sample      
  3) How they improve or solve ?
       1) they generalize the definition 
       2) they group multiple positive and by measuring  similarity in both video and audio feature spaces 
       3) allow to caribrate visual similarities by seeking within-modal discriminative of positive instances   
              
       4) achive siginificant gains on downstreams tasks
  4) the usage of these papers
        1) Action recognition 
        2) sound classification  
  
  *contrasting visual represenation againts audio and vice versa. 
  * CMA help to relate videos and reject false positive which is visual similarity but different in audio space. 
  * they using these group of videos allow to optimize with-in modal similarity in addition to cross modal similarity 
  
  *within-modal discrimination task = predicting which videos clip from the same video clip 


#4. TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION

1) what are the tasks problems or what are the next challenges ?
   - the failure to extrapolate is casued by position embedding        
2) what problem that they solve ?
     acheive longer sentence on inference time than during trainning seen
3) How they improve or solve ?
     changing the position representation method 
       - Attention with linear bias 
     improve 
       - trained on 1024 extrapolate to 2048 achiving the same perplexity on i/p trained on 2048, 11% faster and using 11% less memmory.

  what is their distance ? 
  what is inductive bias ?
     -set of assumption that learners use to predict o/p 
     when given the i/p that has not  encountered  
     eg. the nearest neighbour 

4) the usage of these papers
       Text


#5. Emerging Properties in Self-Supervised Vision Transformers 

1) what are the tasks problems or what are the next challenges ?
     vision tasks 
        L lot of demanding of img both data and computation 
        L their dont exhibit unique properties 

2) what problem that they solve ?
        L collapse mode and exhibit rich visual or semantic information 
3) How they improve or solve ?
       L use data itself to convert pretext tasks 
                      L use the words in a sentence to create pretext eg. GPT3, BERT
                        instead of predict one label per sentence 
                      L 

4) the usage of these papers

      - image 


#6.  Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty 

1) what are the tasks problems or what are the next challenges ?
     - lag fully supervised trainning 
     - robustness problem to adversiarial examples 
     - label corruption 
     - common i/p corruptions
  
* what is out-of-distribution detection on difficult ?
     ans unknow classes
  what is near-distribution outliers ? 
     ans know classes  
2) what problem that they solve ?
     -     
     -  exceeds the performance of fully self-supervised ? 
        - for out-of-distribution detection, they are greater than fully supervised methods
3) How they improve or solve ?
        they called "self-supervised auxiliary learning" -> the img are rotated into 4 different roatations and NN have to predict the type of rotation
4) the usage of these papers
        imgs
---------------
16 sep: progress

#8. ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators 

    1) what are the tasks problems or what are the next challenges ?
         - performance and computation 
         - MLM only learn 15% per examples
         L why in literature they dont trained on i/p token
         L why this work it's matter  -> old work require large compute to performs well on downstream tasks -> they are solved by less computation and get more accuracy 
     
         * they hope that in the future should have included the measuring  the performance of pretext


    2) what problem that they solve ?
         - make more computation efficient -> as they all of i/p token instead of 
         just subset of token  
    3) How they improve or solve ?
         - rather than predicting corrupted i/ p, they predict whehter that tokens are replaced or not 

    4) the usage of these papers
         
         It can use to improve other langauge downstream tasks

    5) what did authors try to accomplish ?
        - In pretrain Tasks -> replaced token detection using  less 1/4  RoBERT & XLNet   
          -ouperforms when using same amount of compute 

***  6) what were the key element of the approach ?
         1)  Instead of masked token, They replace corrupted input token by replacing plausible token sampled from generator network.
         
         2) Instead of predict corrupted i/p, they use discriminative model to classify whether sampled token or original ones. 

       *trainning idea
          L they train the encoder to distiquish whether that token are replaced or not 
          on pretext task 

          L token replaced -> high quality negative examples produced by small generator networks


# 9. Zero-Shot Text-to-Image Generation


    * what is transformer autoregressive models ?
    * why they called zero shot ?
          L guess may be "single stream of data "
    1) what are the tasks problems or what are the next challenges ?
         - drive more data raise generalization  
    2) what problem that they solve ?
         - lost information of img eg. distorted  
    3) How they improve or solve ?
         - They used Autoregressive Transformers and feeding img Text pairs
          without using trainning labels  
           
    4) what can you use yourself ? 
          - tasks after train on img-to-text 
                     L apply to image-to-image 
                           L img to img translation 
    5) what did authors try to accomplish ?
         - better modelling assumptions for trainning on fixed dataset
*** 6) what are key element of the approach ? 
     - they feed the text and image token to complicated models
     
     - complicated models  
           L auxillary loss
           L some side of information -> label of objs and segmentation masking imgs applying when tranning 
                L The approach based on Autoregressive Transformers 

       

# 10. Aggregrating Nested Transformers

   what are the key ideas of this papers 
   
***   - nesting local transformers on non-overlapping img and aggregating in heirachical manners -> 7 min

   - Figure (5 min) 
      - the idea nested transformer Fig 1 
***             * they divied into blocks
                 * feed it into another layers its
                  * guess -> the number of blocks indicate the receptive fields
                  * receptive -> on top of that they can receive from how many blocks
                    
                  * becase of this they can simplify architecture 
        

        what you can use yourself


#11.MultiCQA: Zero-Shot Transfer of Self-Supervised Text Matching Models on a Massive Scale

*** what are the keys idea of this papers  ?
   - They do supervised multitask learning in select source domain
   which incoperate on self-supervised  tasks 
   - the different from state of art they didnt do the most domain similiary instead they wide range of domain make zero shot model perform better than in-domain BERT and previous state 

   what you can use yourself ? 
    - we can use question answering on language tasks
    - less label data or low data privacy 
 
