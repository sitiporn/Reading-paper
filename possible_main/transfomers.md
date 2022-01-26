# Transformers 
---

the convention of embed_dim is the maximum seq len

dempencies
---
 -keys : word position flow each own path but there are some deopencides between these path  

 - allowing to look at other position in the i/p seq for clues to be better encoding 


 attention 
---
 Z = attention{(Q@kt)/sqrt(dk)@V}

 Z ~ each value in Z represent  sum of score of current query towards all seq 

 mulihead attention 
---
 - they concat because they are not related each other so we concat it  

   this is because each it just the representation in different in sub space 

 decoder
--- 
 - the decoder only allow the earlier o/p in seq to feed 
 by masking future positions

 - it works the same as mulitihead-attention except it used query from below decoder keys and value from encoder stack



 Question
--- 
 - why do we need to muliply (q1xk3) and (q3xk1) ? 
   even the value are the same before sum up or just the sake of different query should be repeated to weight value tensor  


 - How residual help in adding Z with X
