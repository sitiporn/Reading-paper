"""
# paper 
    
    * Wall Street Journal (WSJ) portion of the Penn Treebank (4,35)
       * (4) https://aclanthology.org/J93-2004.pdf
       * (35) https://catalog.ldc.upenn.edu/LDC99T42  

    * annotated with Stanford Dependencies(SD) (36)
       * https://downloads.cs.stanford.edu/nlp/software/dependencies_manual.pdf

# Lib 
   * stanza 
       * follow Universal Dependencies formalism accordingg to prof said
   * stanfordnlp -> syntatic parsing   
   * nltk     
       * draw tree -> cant parse syntatic without specifying a Grammar 
       * specifying grammar would be problem   
   * universal stanford dependencies  (Prof suggest)
       * https://nlp.stanford.edu/pubs/USD_LREC14_paper_camera_ready.pdf  
       * emphasized add partially specific relation for some lexicalist instance of compound, preposition, mophological in particular parsing 
"""

# Todo 
# 1. syntatic parsing -> stanza 
#      * make syntatic parsing are correct 
#      * can feed list of sentences
# 2. draw tree  
#      * draw tree from stanza   
#      * draw tree nltk without specifying the grammar   

import stanza

stanza.download('en') 
nlp = stanza.Pipeline('en')

#sentences = ["what expression would i use to say i love you", "the chef who ran to the store was out of food"]

# ex-sentence in paper but parsing still wrong with stanza 
sentences = "the chef who ran to the store was out of food"
doc = nlp(sentences)

print("sentence 1 :")
print(doc.sentences[0].print_dependencies())
print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')


"""
print("sentence 2 :")
print(doc.sentences[1].print_dependencies())
"""
