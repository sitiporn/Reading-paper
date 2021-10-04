
# t-distributed stochastic neighbor embedding (t-SNE)

* Deterministic algorithm
   - given paticular i/p -> algorithm -> o/p   -- always produce the same o/p 
   - by always p

---
Deterministic => (PCA) Principal Component Analysis 

  * on different run 
       i/p  -> system -> o/p 
      
 - current state       next state
            
        i/p = 0/s0  ->  s1
        i/p = 1/s0  ->  s2 

----
Non-Deterministics => t-SNE 

  * on different run 
      
        i/p = 0/s0  ->  s1
        i/p = 0/s0  ->  s2 


*Idea 
   L use t-distribution in computing similarity between pts in low-dimension space 
   L use  heavy-tailed Student-t distribution instead of Gaussian distribution 
                L compute to pts between 2 distribution 

   pij - proportion of similarity Xi and Xj -> prob 

   *** sim Xj and Xi => P(j|i) => Xi would pick Xj as its neighbor 
       L if neighbor are picked in proportion to density under a Gaussian Centered at xi 

    qij - measures similarity between two pts in map yi and yj
     L measure sim in low dim in order to allow dissimilar objs to be far apart in the map
     
     L The minimization of KL-div respect to  yi using grad 
                            L  i/p = yi => model => o/p = xi of KL-div func 

     L result of minimizing 

