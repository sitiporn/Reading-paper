1. what is intent detection or classification ? 
  
  * the automated categorization text data based on intention or aim of that text  
especially in comercial dialog system. 

  
  * ulitmately, every cutomer interactions have a purpose 
   for example they want to refund, Downgrade, unsubscribe, and request more information or even purchased. Then, we get some insighful information of customer in order to response quickly  to imrover sasticfaction and loyalty of customer


2.  Why is this task ? 
   

  * intent detection improve capabilty to support personal customer  or persnal service on large amount of query of users to the system so intent clas
   or detection is a tool 
  
   source: https://monkeylearn.com/blog/intent-classification/


   
3. Problem  existed

   It's related to data scarity 
    
     * in production scenario Problem
       1) small datasets
       2) imbalance between in which each intents and also out-of-scope  sample too 
       3) OOS
           

    why OOS is important 
        L to prevent wrong action of system in production like fig 


    we can think out-of-scope is another intent

    out-of-scope is the input text that are not belong to any predefined classes 


    On right the figure (OOS-detection)
     
     Positive : out-scopes
     Neg : in-scopes 

     1. input : Neg  out: in-scope  -> True Negative  
     2. input : Positive out: in-scope -> False Negative 
     3. input : Positive out: out-scope -> True Positive   





