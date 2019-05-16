# -*- coding: utf-8 -*-
"""
Created on Fri May 10 20:30:42 2019

@author: Efsun Sezer
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May  4 00:22:56 2019

@author: Efsun Sezer
"""
import numpy as np
import dynet as dy
import string
from collections import Counter
from numpy import array
import random
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def process_text(text):
    #replace SPEECH marker with ' '
	 text = text.replace('SPEECH', ' ').lower()
	 return text

def main():  
    filename="./trumpspeeches.txt"
    text = open(filename, 'r',encoding='utf8').read()
    
    clean_text = process_text(text)
    tokens = clean_text .split()
    clean_text = ' '.join(tokens)
    
    
    characters = sorted(list(set(clean_text)))
    n_to_char = {n:char for n, char in enumerate(characters)}
    char_to_n = {char:n for n, char in enumerate(characters)}
    vocabulary_size=len(n_to_char)
    print('vocabulary size: ', vocabulary_size)
   
    inputs = []
    output_labels  = []
    length = len(clean_text)
    seq_length = 1
    for i in range(0, length-seq_length, 1):
         char = clean_text[i]
         label =clean_text[i + seq_length]
         inputs.append([char_to_n[char]])
         output_labels.append(char_to_n[label])
    
    
    inputs = np.array(inputs)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_input = onehot_encoder.fit_transform(inputs)
    
    
   # A feed forward neural network using DyNet 
    model = dy.Model()
    trainer = dy.AdamTrainer(model)
    #number of unique values in list 
    ntags = len(set(output_labels))
    # Define the model
    EMB_SIZE = 64    
    HID_SIZE = 64
    W_emb = model.add_lookup_parameters((vocabulary_size, EMB_SIZE)) # Word embeddings
    W_h =  model.add_parameters((HID_SIZE, EMB_SIZE ))
    b_h =  model.add_parameters((HID_SIZE))    
    W_sm = model.add_parameters((ntags, HID_SIZE))          # Softmax weights
    b_sm = model.add_parameters((ntags))                    # Softmax bias
       
    # A function to calculate scores for one value
     # A function to calculate scores for one value
    def calc_scores(words):
      dy.renew_cg() 
      word=words.index(1)
      h1= dy.lookup(W_emb, word)
      h2 = dy.tanh( dy.parameter(W_h) * h1 + dy.parameter(b_h) )
      W_softmax= dy.parameter(W_sm)
      b_softmax= dy.parameter(b_sm)
      return W_softmax * h2+ b_softmax
  
    
    for ITER in range(200):
      # Perform training
      train_loss = 0.0
      for i in range(5000):  
        temp_input=onehot_input[i,:].tolist()
        input_to_network=list(map(int, temp_input))
        print("------------------sample number is %d------------------" % i)
        tag=output_labels[i]
        my_loss = dy.pickneglogsoftmax(calc_scores(input_to_network),tag)
        train_loss += my_loss.value()
        my_loss.backward()
        trainer.update()
      print("Epoch %d: train loss=%f" % (ITER, train_loss))
      
     
     #Generate Sentence
    input_char='5' 
    generated_sentence=[]
    generated_sentence.append(input_char)
    
    for i in range(50):
        idx=char_to_n[input_char]
        all_index =np.where(inputs==idx)[0]
        random_index =random.choice(all_index) 
        onehot_char_vector=onehot_input[random_index,:].tolist()
        test_input=list(map(int, onehot_char_vector))
        scores = calc_scores(test_input).npvalue()
        predicted_label = np.argmax(scores)
        predicted_char=n_to_char[predicted_label] 
        generated_sentence.append(predicted_char)
        input_char = predicted_char 
    
    print("Generated sentence is \n")             
    print(" %s " % ' '.join(generated_sentence))
    text_file= open("char_level_sentence.txt","w+")  
    text_file.write("%s\n" %generated_sentence)
    text_file.close()
      
if __name__ == "__main__":
    main()


































