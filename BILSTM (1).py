#!/usr/bin/env python
# coding: utf-8

# # Model Architecture

# The model is a sequential model which means that the layers are stacked linearly where the output of one layer is the input to the next layer.
# 
# Embedding Layer: The model starts with an Embedding layer. This layer turns positive integers (indexes) into dense vectors of fixed size. In this case, it transforms the input which is a vocabulary of vocab_size unique words into 100-dimensional vectors. This layer also expects input sequences to be of a fixed length max_sentence_len.
# 
# Bidirectional LSTM Layer: Next is the Bidirectional LSTM layer. LSTM stands for Long Short-Term Memory, a type of recurrent neural network (RNN) that is capable of learning and remembering over long sequences, and is particularly effective for sequence prediction problems. A Bidirectional LSTM, as used here, involves duplicating the LSTM layer so that there are now two layers side-by-side, then providing the input sequence as-is as input to the first layer, and providing a reversed copy of the input sequence to the second. This allows the LSTM to learn both the input sequence forwards and backwards, providing additional context to the network and in theory resulting in faster and more effective learning.
# 
# Dense Layer: The last layer of the model is a Dense layer. This layer is the fully connected layer where each input node is connected to each output node. The Dense layer has 1 neuron and uses a sigmoid activation function. The sigmoid function outputs values between 0 and 1 which is ideal for binary classification problems where the output can be interpreted as the probability of being in the positive class.

# # Hyperparameters

# Several hyperparameters were set before training the model.
# 
# The learning rate was set to 0.001. The learning rate determines how much the model's weights are updated during training. A smaller learning rate might make the model learn slowly, but it can help the model reach a better minimum.
# 
# The model was trained for 10 epochs. An epoch is one complete pass through the entire training dataset. More epochs mean that the model will have more opportunities to learn the patterns in the training data, but there's also a risk of overfitting if the number of epochs is too high.
# 
# The batch size was set to 64. The batch size is the number of samples that are used to update the model's weights at once. Smaller batch sizes can help the model generalize better, but the training process can be slower.
# 
# Max_features is set to 18611, meaning that the model uses the 18611 most common words from the text data.
# 
# The max_sentence_len parameter was set to 1844. This means that all input sequences should have a length of 1844. If a text is longer, it is truncated; if it's shorter, it's padded with zeros. This is needed because LSTM (and other types of RNN) expect all sequences in the same batch to have the same length.
# 
# The Embedding dimension was set to 100, meaning that each word in the vocabulary is represented by a 100-dimensional vector.

# In[1]:


from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd


# In[2]:


df = pd.read_csv('depression_dataset_reddit_cleaned.csv')


# In[3]:


max_features = 18611 # or consider the total unique words in your corpus
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])


# In[4]:


maxlen = 1844 # or consider the maximum length of sentence in your corpus
data = pad_sequences(sequences, maxlen=maxlen)


# In[5]:


labels = np.array(df['is_depression'])


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)


# In[7]:


embedding_dim = 100
model = Sequential()
model.add(Embedding(max_features, embedding_dim, input_length=maxlen))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(1, activation='sigmoid'))


# In[8]:


model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


# In[16]:


history=model.fit(X_train, y_train, batch_size=64, epochs=30, validation_data=(X_test, y_test))


# In[17]:


test_loss, test_acc = model.evaluate(X_test, y_test)


# In[18]:


import matplotlib.pyplot as plt


# In[19]:


plt.figure(figsize=(12,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[20]:


plt.figure(figsize=(12,6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[21]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Predicting on test data
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)

# Visualizing the confusion matrix using seaborn
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




