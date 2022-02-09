# Emergent linquistic structure in ann trained by self-supervison 


 Problem  
---

 * no anotation syntatic dependencies 
      for intent detection datasets ? 


    * they used syntatic dependencies using wall stree journal(WSJ) portion of Penn Treebank annoted with standford Dependencies as Corpus   
 
 baseline 
---
 
    * Normally the word that position neg 2 or to left considered to be head  

 Question 
---
 1) what is this work is about ?   
   show that how large language pretrain model works
      
      * how ?  
          * can learn linquistic structure and syntatic parsing and be able to aproximately reconstruct linquistic stucture   

      * How often the head word of a coreferent metion most attends to the head   
      word of one that montion's antecendents

  
 2) what is combined score 

      
  
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


   * then eqn_3 evaluate that attetion head is expressing a particular linguistic realation by computing how often this particular expression come with the input word     
          

 Experiments
---
 
 * tree structure result from word that have exactly one incoming edges from either another word (syntatic head) a type indicate grmammatical relation 

   * l(wi, wj)  -> 1  
        if wi is the head and wj as dependencies otherwise 0   
 
 * they show prediction fixed-offset baseline  
       * two position two left of dependencies considered to head  

  * Coferernce  compare two baseline 
       
       1) select nearest other mentions as antecendents
       2) using ruled bassed conference system 
          1) full string match   
          2) head word match  
          3) number/gender/person mattch
          4) all over mentions
         
     
 Result 
--- 
  * Normally each head specialize for one dependencies
  * syntatic behaviour is not clearly wrong 
  * according to table 2 relation on Nominal perform better than rule based as Norminal more complicated than other properties like   Prounoun or propernoun 

 Finding Syntax Trees in word represenations  
---
* the vectors of word in each layers embedded the tree systax structure
* tree is discrete structure are not compatible with high dimension like Rd spaces of neural representations

* they use structure probe finding embedding representations is linearly extractable from internal representation     
   * find single distance metric on Rd defined by sytntax tree of that sentence 

 The stucture probe methods 
---
 
  * to test whether tree are embedded in vectors representaions or not  it required common ground between  tree and vectors spaces 
  
  * set of node might be set of word ?   

  * what is undirected tree ? 
     * it may be space

  * metric dT(i,j) between node i and j    
         * defined as number of edges in path between node i and j  
         * given path metrics -> construct tree itself by indicating all i, j with    

         * if dT(i,j) equal to 1 are conntected by an edge  

  * defining distance metrics with tunable parameters 
        
        * L2 distance on Rd can be parameterize by a positive semidefinite matrix
            A subset of S+ dxd
            
        * dA(hi,hj)2 = (hi-hj).T @ A @ (hi-hj)
          a.T @ b = b.T @ a
          let A = B.T @ B

                dA(hi,hj) = (hi-hj).T @ A @ (hi-hj) -> Rd
                          = (hi-hj).T @ B.T @ B(hi-hj) 
                          = (B(hi-hj)).T @ (b(hi-hj) 
                          = ||B(hi-hj)||2


         
          



