# What Does Bert Look At ? An Analysis of BERT's Attetion 


 Adopt 
--- 
  * what is specific language feature that they can use to help them discriminate intent  
  
  * in related work -> one line of research examine the outputs by given carefully select i/p sentence 
      * can we change examine the attetion head by chosen the intent exaple carefully both positive and negative pairs  to understand model performance on intent detection datasets by evaluate attetion head on linguistic structures  or dependency parsing 

      *  identify syntatic, positional rare-world-sentive attetion head that effect the each intent   

   
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
 * How much each weight to each word when computation next word based on current word
 * How compute Weight of attetion two directions are important ?
   > this is because they can show how much likely to be head
      > by Wk or Uk can tell likely direction 


Eqn
---

 * alpha_ij = exp(qi@kj)/(sum_l(exp(qi@kl))
  * weight of each word to all other word in the sentence 
  * ith ~ current word   

  * calculate weight of current word by sum up all value of weighted values particular wod with all other value vectors 
      
      oi = sum_j(alpha_ij@ Vj)

  * p(i|j) ~ probability of word i being word syntatic head of j 
     * combine both direction candidate head to dependents as well as dependents to head
     * wk and uj using standard supervised learning on trainning set
     * (vi ⊕vj ) concate glove embeddings of two vectors are fixed once learning so only W and U are learned  
     
     * produce sensitivity weight for particular head



 key question
---
    * how important of other token in sentence when predict next representation of current token  

    * they used gradient of loss with respect to each attention weight  
         * from their experiment can be discussed that  oviously started at layers 5 
    changing attetion whether more or less wouldnt be affected loss or o/p of model as gradient is still low  which answer to their assumption when atteion head no-op all token would be attend to [SEP]  

    * raw of grammar the head can have multiple dependents or can modified by many dependents while one dependent can related to only one head 
        * in example Head they dont follow direction convention of depency grammar      
       but they show related btw dependents and head  
        
        * in Fig the red represent head which is able to attend to many modifiers    

    * from Fig 5. even BERT are not explicitly trained on tasks like depency parsing but self-supervised can make them  learn this kinds of task well according to the head that do specific function for language  lead to easy to illustrate how sensitivity  of attention weight toward output of the modl

    * normally, they encouraged different attetion heads to have different functions but result show that they are closely and redundance so they used attetion dropout in training caused some weight lead to zero 

 finding
---
 
 model get indirecly supervision from large data set can sensitive produce languge hierachy 

 Result
--- 
 
 * on table 3. attetion glove provide syntatic parsing much more information than vector representations
 



