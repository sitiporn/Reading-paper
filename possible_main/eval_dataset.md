# An Evaluation Dataset for Intent Classifcation and Out-of-Scope Prediction  


 Adoption
---
   
   * can we analyze more wrong prediction of out-of-scope sample still falls into in-scope classes ?    
      * what is word that model pay attetion to, to make this prediction belong to in-scope classes 

   *  what is the liguistic structure look like when the model can be able to know that this out-of-scope samples
   
   * in production scenario Problem
       1) small datasets
       2) imbalance 
       3) OOS

* Why is important  
   * prevent wrong action in system in production and also for future developement

* Problem 
   * out of scope is happen in dialog system in production that this important



 Keys
---
 * out-of-scope threshold
      L is high valaidation score across all intents -> look out-of-scope as another intents



 Metrics
---

  1) they used acc for all intents
  2) for out-of-scope 
       * pos-> out-of-scope 
       * Neg-> in-scope 
     
     when sys get out-of-scope queries 
        it predict in-scope and provide wrong actions  
     which is predict FN
        
     i/p (out-of-scope) -> model understand -> identify that in-scopes  (FN)
     i/p  (in-scope) -> model understand -> identify that out-scopes (FP) 
           L then system just return that can you type again which is better than when model get out-of-scope and provide wrong actions

     * Precision - How much positive are correct 
     * Recall - How much positive are indentified correctly 


 Result
---

  * Trend of acc of data in (small and imba) slightly worst than Full which is  opposites  to OOS data the recall higher than Full   
      
      * in in-scope training data decrease but in out-of-scope remaining constant  but recall still increase this mean that if we increase samples recall would be increase   
      

