Survey Attention: 063-3109191

what is Attention:
  L focus , selective of cognitive process while ignoring other perceptual information

:definition:
  “Attention implies withdrawal from some things in order to deal effectively with other"

Computational attention systems
  3 features: intensity, orientation , color

Problem:
  1. from src sentence to fixed legnth of i/p vect 
  2. when i/p increase deteriorates  

What Attention solves 
1.elimate fixed size context vect's information


RNNSearch -> Attention
L not encode entire i/p sent into a single fixed-length vect  instead encode i/p  sent into seq of vect  


Attention Mech 
 L allow extra information through to network eliminate fixed size context's vectors adaptively outperform classis encoder decoder frameworks with longer sentences


# 2.1 Attention

    BiDAF
    L multi states hierachy process to question answering 
    L do not sum context paragraph into fixed length vector
    L  

  what do they mean previous stage attention ?
       attention layers   
        L learning attention between  query and context
       
       modeling layers 
        L

    Hierachical Attention Network(HAN) page5
      represent the sentence then aggregrate into docs
      two level attetion focused on individuals 
        L word 
        L sentence 
        
    Ptr-Net[54] 
      L modifies attention mechanism to represent variable- length dictionaries
      L question and document via *pointing 
      L act as pointer
      L [54] and [55] used to keep track things summarized

    FusionNet[56]
      L "history of word" concept to characterize word-embedded level to semantic levels
      L  above concept transforms abstract representation 
        L fully-aware multi-level attention mechanism and an attention score-function

# 2.2 
    
