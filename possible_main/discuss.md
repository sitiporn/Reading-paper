# Supervised Contrasive Learning loss


neg log likelihood
---
 
 - for each pair -> - log(x) 
  
    1. neg log the more i/p  -> the lower loss
    2. Expected, we have to maximize x  -> maximize similarity from the same class 
    3. Expected, we have to minimize divider of x by have low sim between current samples with all other in the batch 
  
    why bottom terms factors would not be negative pairs which try to lower similarity ?   



intent classification loss  
---

 - neg sum_j sum_i log(P(Cj|ui))
  
   j - belong to class  
   i - belong to samples

   P(Cj|ui) - Softmax(Wx+b) ; logits ~ Z  

   
   so we have to maximize probability of class pj to the right class  y_true
   and minimize the probabilty of other class 

 - Traditional cross entropy loss 

    -- cross entropy minimization  
        compare distrbition q with fix references distrbition p
        cross entropy and KL-div are identicali up to additive constant (since p is fixed)
       
        - when q is fixed references and p are optimized close to q  as possible subject to some constaint !!! in this case two minimization are not equivalent 


 - ML world
       
       pi - is the true label -> p E {y,1-y} for bin
       qi - given distribution by predicted value -> q E {yhat, 1- yhat}





