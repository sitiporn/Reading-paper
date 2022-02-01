# What Does Bert Look At ? An Analysis of BERT's Attetion 


 Adopt 
--- 
 what is specific language feature that they can use to help them discriminate intent  
 

Ways of analysing 
--- 
 1) specific i/p and notice o/p 
 2) see vector represenation lool like 
 3) *look at attetion head activate

 Guess
---
 
  show each attention part response to each i/p inside networks 


 question  
---
 How much each weight to each word when computation next word based on current word


 Eqn
---

 * alpha_ij = exp(qi@kj)/(sum_l(exp(qi@kl))
  * weight of each word to all other word in the sentence 
  * ith ~ current word   

  * calculate weight of current word by sum up all value of weighted values particular wod with all other value vectors 
      
      oi = sum_j(alpha_ij@ Vj)

 key question
---
    * how important of other token in sentence when predict next representation of current token  

    * they used gradient of loss with respect to each attention weight  
         * from their experiment can be discussed that  oviously started at layers 5 
    changing attetion whether more or less wouldnt be affected loss or o/p of model as gradient is still low  

    * raw of grammar the head can have multiple dependents or can modified by many dependents while one dependent can related to only one head 
        * in example Head they dont follow direction convention of depency grammar      
       but they show related btw dependents and head  
        
        * in Fig the red represent head which is able to attend to many modifiers    

