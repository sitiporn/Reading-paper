Supervision: 

 - they can learn visual representations using these representatation 

What is self supervised learning 

-follow supervised labeled obtained semi-automated , without
human i/p
     Lwhat is semi-automated
- a part of data are hidden the rest are visible the aim is to predict the hidden data 

How self-suprevised is differs from supervised learning and unsupervised learning ?

- SP humans provided labels
- UP data sample without supervision 
- Self-SP -> they can predict any part of data hidden by using obseved data -> make use of variety of supervisary signalsaccross co-occuring modlaities all without relying on lables


#Q and A

What they are Good at ?
 -we dont know all possible categories that we want recognize
  
Common sense is the dark matter of intelligence
 L inject abstraction to laguage


Are Augmentation knowledge via the back door ?
 - sort of cheat but the data are free and random
 
Conceptual abstraction in Vision
 - categories dont exist
        L notion hard -> context dependent impossible
        
- CLIP models abstract 

 - notion categories are not exist
 eg. if you hungry person we really care ->cartoon banana , banana
         then cant eat
         context dependent -> depent what we want to do
         really to discover
         self-supervised -> relaxed space

         
#why self supervison work
   L augment same img 
   L the model learn this img are different
   
   give good signals is this the same or different

   random initialize convnet 
     boostrap by data augmentation 

   cheat a little bit
    making inductive bias , data augmentation
    
   cant naming thing but they can tell which are same img or different

#Semantic
  - it just similarities 
  - 1 million standadt img bias sample 
  - slight gap -> samantics of visually similar
  - drawn from inductive prior 
  - categories are not atbitary defined
  - can do pure on Augmentation 

#What do we know of data is -> random walk of picture of the wolrd
- a lot of augmentatio overfit to imagenet of object centrics img
- another souce of hidden injection into model these are not completely random pictures

# get rid of data augmentation what if remove augmentation
we could have uniform distribution is it even better

- should work in theory better but hard to evaluation

# Billion imgs 
    L how important quality control when collecting data
         L randoms image
    L problem -> inductive bias -> object centric
    L it doesnt matter
# Amazon mechnical terk
    L pop up two imgs there question
       1) what are these two image are related
       2)  what are these two objects in image are related
       3)  what are  context of these two image are related

# Achitecture what are the big idea and why ?
      L paper in 2019 increase the channel width of resonat 
      L king of transer to self-supervised

      L contrasive 
             momentum encodees -> small bathch size
             eg. Dino  -> easy or cheaper to create teacher models
             eg. Effiecientnet

      L main teafcher self-supervised
      L the ituition learn similarity to fined-grain for braod categrizartion tasks  
      L the projectors 
         L do particular tasks and dont corrupt features

#Mode collapse
  L How to avoid trivial solution 
    L constrasive methods  -> define positive same imgs negative diff imgs
    L siamenease net 
        L work on contrasive 
        L momentum encoder just effieciency

# How SSL is different in vision over language ?
    L what coud NLP learn come from computer vision
        contrasive para and non softmax know exactly word one image dynamic changing
    L  contrasive eg audio signal and img model can how are they relate

# The Dark matter of self supervised 
    
   - latent models can capture uncertainty inside the latent variable 
   -  basicly introduce the input as stoachastic  one way is to introduce i/p stochastic  
   - predict multiple type of o/p because uncertainty
   - latent variable itself capture everything
   - the engegy you minimize at inferece time the loss you minize at training time 

#Misleading Assumption 

  
  - Self supervised is more accpeted term than unsupervised is used feedbacks more than standard supervised and reienforcements 
         L supervisory signals
             L generated from data itself
  

#Question 

co-occuring modlaities all without relying on lables ?



summarize blog

SP:

- massive amounts of carefully labeled data.
- bottle neck for building generalist models can do multiple tasks
and acquire new skill without massive amounts of labeled data

self-SP:

- generate label out of the data itself 

eg.1 In BERT   -> i/p This is a cat , 
                  L cross out This is _ cat   -> X
                    L a  -> label

- I/p correpted data -> o/p -> uncorrupted
- create label from the data
- X -> Y ; How could we create data points by given  data points

How they trained

opt1 use the past so far and predict the next data points or seq data
    L eg GPT-3  word so far and predict the next word or few words 
    L videos also the same in time space

opt2 leave away some part in the middle
    L BERT

opt3 General 
    L Vision 
       L not only leave some frame in middle

  Past ->  Future
   X1       
   X2
 
   X3
   X4

 Past -> happy ? 
      L let's say X3 , X4 are imporatant or relevant feature 
then X1 and X2 relearned to be overwrited which are not important then they 
can apdapt rapidly to the ones that are more important -> witout this in supervised

Problem
                   NLP   CV
1) dimensionality  ok  Not ok
                 
2) uncertainty   ok   Not ok

Solve
  L Contrasive Learning 

Energy Based model
  L Energy based Function -> how well x and y fit together  
     L crop patch from same similar image if they know they are similar
       L they can learn good representation -> High inner product
       L img1 img2     
            L without contrasive falls into collapse  (img1 ,img3) 
                  L >  make same representation inner product it high 
                              L sastify constrained -> 
                 
                  L Energy is not higher for nonmaching (X,Y) than it is  maching then
Solution
 1) contrasive
       
 2) regularazation

siamese network

1) share weight of encoded
2) multiply -> similarity   Low -> similar
                            High -> diffentiate



Contrasive energy-based SSL

- they sample from blue dot (compatible)  and green dot(uncompatible)  on graph

- crop on mulitiple pair img that are similar

eg. BERT -> contrasive push up correct class , push down uncorrect

Predict Architecture
  - what joint embedding  
  - z -> come from domain   move Z -> move along mainifolds
  - sample z 
  - deterministics 


