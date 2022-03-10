# Discriminative Neeartest neighbors Few shot intent detection 

Term
--- 
 out of scope detection -> out of distribition detection 

Problem of previous work 
--- 
 * large scale pretrain model stuggle with  out-of-dis detection 
 * they strong in-domain predcition 

why intent is important ?
---
  it is the core component in taks diaglog systems
  "different system have diff intents" so that's the reason why few shot is important

* methods
---
  use dnn with deep self with self-attention 
  utilize NLI  


what they acheive ? 
---
  accurate in-domain and oos detection acc than roberta-base 
and embedding nn  

  10 shot result comapative with 50 shot keeping inference time constant 
  acheive high acc and detect unconstrained users intent that are out of scope in a system


what their idea ? 
---
  * predict that i/p utterances belong to the same class of a paired examples
  * expect to free model   


 Methodology look like 
---
   * model inter-utterances relationship in their nearest neighbors classification
   * i/p -> model -> binary o/p 
   * using pairwise example with seamless tranfer of NLI

3.1 Deep pairwise Maching Function 

   - find best macthed utterance from training set E  given u  
    1. embeded each data into vector space  
    2. use off-shelf distance to perform sim search 

  * problem text embeddign dont discriminate OOS examples well enough  
  * to mitigate above problem using S(u,eji)    

    step 1.
     using  h = BERT(u, eji) subset of Rd 
    by changing i/p format : [[CLS] ,u, [SEP], eji, [SEP]] 

    step 2.
     
     S(u,eji) = sigmoid(W@h + b) 

     the idea mapping encoding and matching func into deep self-attenion


  * capturing queries(u) and documents(eji) in documents retrieval 


3.2 Discriminative Training 

   - Pos pairs intent class: (eji, ejl) all possible ordered pairs with in the same intents
    i not equal to l

   - Neg considered all any possible pairs accross two any different class

   - S(u,eji) should be closed to 1.0 if u and eji are belong to same classes otherwise 0.0 trained by using binary cross entropy loss 

3.3 Seamless Transfer from NLI 
    
    let model to learn relationship between two utterance instead of explicitly modeling intents 

    - using transfer learning another inter-intent relation tasks

    - taks whether the hypothesis are entailed by a premise sentence
       - binary classification
       - Pretrain model with NLI task
       - No need to modify model architecture since task format is consistent


    Procedure 
      - Pretrain model wiht NLI task 
      - fine-tune NLI model wiht intent classification training examples check on 3.2 section



 Backgroud knowledge
---

2.1 Task: Few-shot Intent Detection 
 - u : Utterace every in Dialaoug systems task
 - I(u) = C  : user' intents 
 - C  predefine intent class, or is categorize as OOS  
 
  *any utterances can be OOS as long as they are not falling in any class

 - i- denote training examples 
 - j-th class
 - eji ∈ E- set of examples 

2.2 Multi-Class Classification 

 - apply threshold-base stategry to cope with intent classification and OOS detection 
  to softmax o/p
 
  * N the number of intents

 - p(c|u) = softmax(Wh + b) ∈ RN  trained by cross entropy with ground truth intent label
 
 - if p(c|u) > T  * T - [0,1.0] is threshold to be tuned
   otherwise I(u) = OOS

2.3  Nearest Neighbor classification

    - Classify an input into the same class as the most relevance training examples  

    - I(u) = class(arg max eji S(u,eji))
    - S - function estimate relevance score between u and eji
    - To detect OOS they use uncertainty-based strategy 
    - if  greater than relevance score threshold, and otherwise OOS 


