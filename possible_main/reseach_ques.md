# Research question 

## Problem statement 


1) problem of they are the diffent intent
   - on OOS dataset -> on diffent intent some have sematically similar utterances -> they cannot classify 

2) what if dialauge system doesn't support 
   - make them -> doesnt support
   - create new intent or new label  for them 
   - they dont have intent 


# Detail of intent detection tasks
  
  query - ask question about something in order to express doubts about it 
  or check its validity or accuracy 


   what is intent classification ?
     - categorize text into intent or customers goal what they want accroding to text information into categorization eg. customers unsubscribe, downgrade.

   How does intent classification work ?
      - use text samples to train the models text should be tagged Then,the model should be able to learn figure it out  that this text in which  associated  with class ?  
      - eg. "I tried to make a purchase through the site but I dont know where to start, could you help me out  "  
      * tagged-> interested       
      - models can be automated classify from user's text after trainning 

   Why intent classification useful ?
      - intent classification is useful when we have to with large number of quries of users which personallize reponsive services which is call customers centric  


   How to get start with intent classification ? 
      1) create your classifier
      2) Define type of your classification
            L topics classification 
            L statiment analysis 
            L Intent classifcation 

      3) import data 
      4) Define 'Intent' tags 

      
  *ref- https://monkeylearn.com/blog/intent-classification/


# Research question of literater review 
  
   OOS: the users request are not be supported by testing daiglog system  
    
    L OOS rately over-lap eg. requesting Tv shows in banking systems
    L separating ID-OOS  eg. requesting banking services that support by banking systems
    
     

    1) Are pretrannning Transformers robust in the intent classification transformers ? 
    2) Is the schema with both stages necessary?
    3) Is the training sensitive to hyper-parameters?  
    4) * whether self-supervised contrasive pre-trainning only target intent datasets benefits 