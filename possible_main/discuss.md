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
 
   so we have to maximize probability of class pj to the right class  y_true
   and minimize the probabilty of other class 
