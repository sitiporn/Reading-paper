---
1. Are Pretrained Transformers Robust in Intent Classification? A Missing Ingredient in Evaluation of Out-of-Scope Intent Detection

 ** guess -> pretain transformers didnt robust in Out-of-Scope Intent 
            
 what is out of scope mean ?
  L the dialouge system are not supported   

 * what did authors try to accomplish ?
     L the problem in pretain models that are in-domain but out-of-scope 
            L ID-OOS examples
            L general out of scope  ** -> fined grained few-shot intent detefction tasks
 * what were the key element of the approach ?
 * what can you use yourself ?
     - we can investiage more on OOS detect task
 
   *** further work -> pretrained models are not robust on ID-OOS
    examples, and both the OOS detection tasks are
    challenging on the scenario of fine-grained intent
    detection.
  
 Out-of-scope (OOS) -> The request of user dont be expected or supported by the tested dialog system 
  
   * import to improve OOS intent detection while keeping in-scope in few-shot  
   
  OOS examples -> daset dont belong to any of known intent classes 

   * their reseach question -> Are pretained Transformers robust in intent classification 
   
   * ID-OOS , OD-OOS , ! ID = in-domanin OD = out-domanin 

  the problem -> pretained model are less robust on ID-OOS than in-scope and OD-OOS exameples; 

  subtask may be -> improve on OD-OOS and ID-OOS fined grained on few-shot detection 
  
  what is masking keywords shared among confusing intents
  
  why OOS examples can be considered as out of domains ? 
  
 CLINC’s OOS <-> out-of-domain 


 OOS examples and the original Banking77 dataset


 OOS-OOS problem -> they cannot evaluate capability to detect OOS intents with same domains

 what are the targeted domains ? 

 OOD-OOS: out-of-domain OOS . General out-of-scope which are not supported by dialouge systems,also called out-of-domain eg. requesting NBA TV show service in banking system 

 
 ID-OOS: in-domain OOS, out-of-scope which are more related to in-scope intents <- challenge ! 
 eg. requesting a banking serviced that is not supported by the banking system. 

 :CLINIC dataset:
  - 10 domain select 15 intent for each
  - randomly select 2 domain from 10 domain to evaluate
                    L Banking
                    L Credit cards


 :The BANKING77: 
  
  - fined-grained single banking domain intent dataset with 77 intents
  
  - exclude OOS examples


  ** use to two above dataset

     1) to detect OOS detection task 
     2) conduct the evaluation across dif domains and single larged fine-grained domain. 
  --- 
  2. A Simple Language Model for Task-Oriented Dialogue

     * Joint probability cannot be used to determine how much the occurrence of one event influences the occurrence of another event

     * NLU -> belief state tracking
     
     * DM  -> diaglouge management for decided which action to take based on those belief state tracking 
     
     * NLG natural language generation (NLG) for generating responses

     * beliedf state and action decision are generated rather than retrieve 
     
     * what do they mean "setting closet to testing" diaglouge system
     
     * summary (SimpleTOD)  
         -> leveraging GPT-2 for task-oriented diaglouge 
         -> show good connection between NLU in open domain require high quality language models and understanding require for a full task-oriented dialouge system


     what is inherent dependencies between the sub-tasks of task-oriented dialouge 
           L guess link all tasks have dependencies by optimizing end-to-end.
     
     
     what are they trying to accomplish ?  
       - SimpleTOD generative model for diaglogue state tracking, action decision, and response generation to achive state of art 

       - robust for dialouge state tracker in the presence of noisy-labeled annotations

       
       
     what were the key element of their approach ?
       - they recast daiglouge system as a simple language models -> can solve all subtasks by unified approach 
       by using multitask maximum likelihood trainning 
       
       -- they predict one token at a time by using previous token to predict next token 
       and also use generated belieft state  to qeury to get the db response 
       until they predict end token 

    what can you use yourself ?

     ---

     what is joint goal acc for diaglouge state tracking ? 
     what is casual language model ? < - why we need them 

3. TOD-BERT: Pre-trained Natural Language Understanding for
Task-Oriented Dialogue
    
    * what is Task-Oriented Dialouge ? 
       - the converstation between bot or assitant and human 
       - help users specific goals 
       - Focus on understanding users, tracking state, and generating next actions.
       - Fewer turns the better 

    *  Key Terms
      
       - Domain ontology : a set of knowledge structure representing the kinds of intentions the system can extract from users      
         
         ** may be I should use GNN to relate information structure of intentions
         
       - Domain: a domain consist of a colletion of slots.
       - slot: each of slot can take a set of possible values.
    

    ---
    * what are they trying to accomplish ?  

    * what were the key element of their approach ?
    
    * what can you use yourself ?



--- 

4. BERT for Joint Intent Classification and Slot Filling
 
    Lol -> However, there has not been much effort on exploring BERT for natural language understanding. 

    BERT: faticilate Pre-tranning models on large-scale unlabeled corpora 
    
    what is slot filling 
      - identify running diaglouge different slot, which correspond to user' quries.
        eg. when users ask "near by restaurant" 
                Key slot -> location , preferred food 
                  L required for diaglouge system to retreive appropriate information  

    * the problem also lack of human label data for NLU 
      result in poor generalization 
    


    ---
    !!! next time elaborate more slot filtering and intent classification

    * intent classification & slot filtering are essential tasks  for natural language understanding 

->   :on wikipedia:
      * intent classification -> predict intent of query
      * slot filling -> extract the semantics concept
                L location , preferred food  
        
        eg1. "near by restaurant" 
                 intent: find restauran
                 slot filling: what kind of food ? , where are restaurant ? 
->    :on papers:

      * intent classification -> predict the intent of label yi
      * slot filling -> a seq of labeling task that tag the i/p word seq
 
     --- 
     (X1,...,XT) ->  BERT ->  (h1,...,hT)

      yi = softmax()

      what do they means by final hidden states of other tokens
      (h2,...,hT) ?
        
       L  but they clasifiy over slot filling labels
      
      what is WordPice tokenizer ? 
        - a subword-based tokenization algorithm 
        - eg. linear = li + near or li + n+ea+r 
      
      what is semantic roles labelling 

      why they use hidden state corresponding to first sub-token as i/p to classifier ?  

      ---
      sec3.3
         
         * slot label prediction depend on surrounding words 

         :model:
   
   g1 -> joint intent  classification and slot filling models: seq based joint models
         using BiLSTM, attention-based models and slot-gated model
   
   g2 -> joint BERT 

       * adding CRF for modelling  slot lael dependencies on top of joint BERT model.
       CRF : conditional random fields
        L on the seconds datasets called "ATIS" 


     ---
    :report prof:
    *  what did authors try to accomplish ?        
         -addressing poor generalization of Traditional NLU models


    * what were the key element of their approach ?
        - they joint both intent and slot filling together instead of separately 
    
    * what can you use yourself ? 
        - evaluation on proposed approach on large scale and complex NLU datasets and exploring other knowledge with BERT 


    * IDEA from papers ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators
       what were the key element of the approach ?
         1)  Instead of masked token, They replace corrupted input token by replacing plausible token sampled from generator network.
         
         2) Instead of predict corrupted i/p, they use discriminative model to classify whether sampled token or original ones. 

       *trainning idea
          L they train the encoder to distiquish whether that token are replaced or not 
          on pretext task 

          L token replaced -> high quality negative examples produced by small generator networks
    
    *  Out-of-scope (OOS) -> The request of user dont be expected or supported by the tested dialog system 
    
    * one of possible tasks woulbe ->  Out-of-Scope Intent Detection and slot filling 

    * On papers "Are Pretrained Transformers Robust in Intent Classification?

    ---
 
 5. (0, 2021) Few-shot Intent Detection via Contrasive Pre-trainning and Fine-Tuning 
    
    what is self-supervised contrasive pretainning ? 
     

    intent detection -> aimming to identify intent of user utterances
        L key task dialog system 
    
    guess -> the reason that we have to do few shot is because we cannot keep 
    all intent sample as our data and also rare and expensive 
    
    why they train on relative intent pretain dataset or target intent dataset
    

    * verify -> the term  semantically similar intents 
    
    * many intents in dataset are similar -> limited examples
    
    problem 

      1) on many fined grained intent
      2) **especially semantics similar intents
    ---

    * what did authors try to accomplish ?        

    *  what did authors do ?        
        
        1) self-supervised Contrasive Pretrainning models to discriminate the semantically similar utraunes 
        
        2) use Supervised Contrasive learning to pull the same intent together and push difference intent far apart 
        

    * what are the key idea of this 
        
        1) Data Augmentation utilizing VAE, GPT-2 
              L doing just augment it's not enough wheh we have to fined grained on lot of intent and also has a semantically similar utterances
              
            


        2) Task adaptive pretain models 
              L utilizing self-supervised contrasive pretaining to discrimate semantically similar uterances with label on intents dataset 
              
              L then, supervised-contrasive learning  and fewshot intent detection
                they used self-supervised to pull the similar intent together but diff far apart

        
        
    * Experiment result
        
        - their  model outperforms all datasets CLINIC150, BANKING77, HWU64
      and variance also low


    * what can you use yourself ? 





* How can I use this ?
    * why I must to use this ? 
    * when wil I use it ? 

