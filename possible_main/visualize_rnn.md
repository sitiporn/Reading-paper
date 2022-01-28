 Visaulizing and understanding Recurrent Networks
--- 

1.Why they are important ?
 
 -  no source of interpretation how they worke inside
 - if they are interpreable utize in designing better architecture 


what they did ?  
---
 
 - the model work on character levels which is interpreable or be able to examine what LSTM predict  on long range dependencies
 - something finite horizontal n-grams models trace improvement to long range dependencies 




 Performance Comparisons
---
 - when the n small the performance of  model are identical but when n bigger the n-NN become overfit which N-grams better than n-NN

 - but when more than 20 grams their best rnn models perform better which rnn are good at long range dependencies 


 Erorr Analysis 
--- 

 - RNN is interested in carriage return significantly better than 20-grams models this characters used long-range dependencies in considering 

 - this can tell that LSTM good at considering long range dependencies


 Trainning dynamics 
---
  -  LSTM can imporve consider long range dependencies overt time 

  according to Sutskever he reverse source of sentence the model can model short dependencies first and then improve over time   

  - inverstion of sentence can clearly spot stage of improving of dependencies over time compare to without inverstion  ?  


 4.4 Eroor Analysis: Breakdown failure cases
--- 

 - considered charactors as error if the prob abality of previous word are lower 0.5 

 - error can elimnate by increase data samples 
 - error can discriminate by bigger model too   
