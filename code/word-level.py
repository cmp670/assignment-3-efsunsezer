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
from numpy import argmax
import random

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def process_text(text):
    #replace SPEECH marker with ' '
	text = text.replace('SPEECH', ' ').lower()
	# split into tokens by white space
	tokens = text.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	return tokens

def main():  
    filename="./trumpspeeches.txt"
    text = open(filename, 'r',encoding='utf8').read()
    clean_text = process_text(text)
   
    min_count = 2
    unknown_token = '<unk>'
    word2index = {unknown_token: 0}
    index2word = [unknown_token]
    
    filtered_words = 0
    counter = Counter(clean_text)
    for word, count in counter.items():
        if count >= min_count:
            index2word.append(word)
            word2index[word] = len(word2index)
        else:
            filtered_words += 1
    
    vocabulary_size = len(word2index)
    print('vocabulary size: ', vocabulary_size)
    print('filtered words: ', filtered_words)
  
    step = 1
    maxlen = 1
    inputs = []
    output_labels = []
  
    for i in range(0, len(clean_text) - maxlen, step):
        current_word= clean_text[i]
        next_word = clean_text[i + maxlen]
        if set((current_word,next_word)).issubset(word2index) :
            inputs.append(word2index[current_word])
            output_labels.append(word2index[next_word])
            
   
    inputs = np.array(inputs)
    outputs = np.array(output_labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_input = onehot_encoder.fit_transform(inputs.reshape(-1, 1))
    onehot_output_labels=onehot_encoder.fit_transform(outputs.reshape(-1, 1))
    
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
    def calc_scores(words):
      dy.renew_cg() 
      word=words.index(1)
      h1= dy.lookup(W_emb, word)
      h2 = dy.tanh( dy.parameter(W_h) * h1 + dy.parameter(b_h) )
      W_softmax= dy.parameter(W_sm)
      b_softmax= dy.parameter(b_sm)
      return W_softmax * h2+ b_softmax
    
    for ITER in range(1):
      # Perform training
      train_loss = 0.0
      #len(onehot_input)
      for i in range(1):
        
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
    input_word="you" 
    generated_sentence=[]
    generated_sentence.append(input_word)
    
    for i in range(50):
        idx=word2index[input_word]
        all_index =np.where(inputs==idx)[0]
        random_index =random.choice(all_index) 
        onehot_word_vector=onehot_input[random_index,:].tolist()
        test_input=list(map(int, onehot_word_vector))
        scores = calc_scores(test_input).npvalue()
        predicted_label = np.argmax(scores)
        predicted_word=index2word[predicted_label] 
        generated_sentence.append(predicted_word)
        input_word = predicted_word      
        
        
    print("Generated sentence is \n")             
    print(" %s " % ' '.join(generated_sentence))
    text_file= open("word_level_sentence.txt","w+")  
    text_file.write("%s\n" %generated_sentence)
    text_file.close()
if __name__ == "__main__":
    main()


































