# Emergent linquistic structure in ann trained by self-supervison 



 Question 
---
 what is this work is about ?   
   show that how large language pretrain model works
      
      * how ?  
          * can learn linquistic structure and syntatic parsing and be able to aproximately reconstruct linquistic stucture   
  



 Keys
--- 
 * language understanding require -> structure are not looking explicitly 
 * normally humand label on treebank structure
 * what the model can captures 
      1. word class (part of speech)
      2. syntatic stucture (grammatical relations or dependencies) 
      3. coreference (which mentions of an entity refer to the same entity) eg. "she" refers back to Rachel
      
      * which is useful to multi-dimentional supervison signal 

  * W argmax_i alpha(w,h) j_i -> prediction   
  * Wj current position  
  * l ~ (wi,wj) if wi that we get from prediction have relationship with wj 
          then it would be corret if not is zero 

          * so precision of h count all correct relation compare to all real relation then we can get accuracy of this prediction of h  

          * the W argmax_i alpha(w,h)j_i 
                * alpha is just ditribution over head attetion  
                * alphaj_i attetion weight toward pos i from pos j 
                * from this eqn we will get the most attended-to of word i from word j  


   * then eqn_3 evaluate that attetion head is expressing a particular linguistic realtion by computing how often this particular expression come with the input word     
          
 Methods
---
  

