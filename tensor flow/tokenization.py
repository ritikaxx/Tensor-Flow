#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing tensorflow and necessary libraries
import tensorflow as tf
from tensorflow import keras


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[2]:


#training data

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]


# In[3]:


#Tokenization is essentially splitting a phrase, sentence, paragraph,
#or an entire text document into smaller units, such as individual words or terms.
#Each of these smaller units are called tokens.

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index


# In[4]:


#NLP Sequencing is the sequence of numbers that we will generate from a
#large corpus or body of statements by training a neural network.
#We will take a set of sentences and 
#assign them numeric tokens based on the training set sentences.

sequences = tokenizer.texts_to_sequences(sentences)


# In[5]:


#padding sequenced data to the max length of the sentence for working with sentences of differet length

padded = pad_sequences(sequences, maxlen=5)
print("\nWord Index = " , word_index)
print("\nSequences = " , sequences)
print("\nPadded Sequences:")
print(padded)


# In[6]:


#testing data 

test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

#sequencing

test_seq = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequence = ", test_seq)

padded = pad_sequences(test_seq, maxlen=10)
print("\nPadded Test Sequence: ")
print(padded)

