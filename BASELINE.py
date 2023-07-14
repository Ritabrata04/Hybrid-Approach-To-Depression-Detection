#!/usr/bin/env python
# coding: utf-8

# # Model Architecture Description

# The baseline model is a simple feed forward neural network,which consists of four layers:
# 
# Embedding Layer: The first layer is an Embedding layer, which is used to convert each word in our input sequences into dense vectors of fixed size, effectively creating a word embedding. This specific embedding layer transforms an input vocabulary of max_features unique words into embedding_dim-dimensional vectors (in this case, 100 dimensions), for input sequences of length maxlen.
# 
# Flatten Layer: Following the Embedding layer, a Flatten layer is used to transform the output from 2D to 1D. This layer does not have any parameters; it only changes the shape of the input, and is necessary to connect the Embedding layer with the Dense layer that follows.
# 
# Dense Layer: The third layer is a Dense layer (fully connected layer) with 32 neurons and a Rectified Linear Unit (ReLU) activation function. It transforms the flattened input by applying a linear transformation and then a ReLU activation, which adds non-linearity to the model.
# 
# Output Layer: The final layer is another Dense layer with a single neuron, used for binary classification. It uses a sigmoid activation function, which transforms its input into a value between 0 and 1. This can be interpreted as the model's confidence that the input sentence is positive (in the context of binary classification).

# In[23]:


import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report


# In[3]:


df = pd.read_csv('depression_dataset_reddit_cleaned.csv')


# In[4]:


max_features =18611
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])


# In[5]:


maxlen = 1844
data = pad_sequences(sequences, maxlen=maxlen)


# In[6]:


labels = np.array(df['is_depression'])


# In[14]:


history=X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)


# In[8]:


embedding_dim = 100
model = Sequential()
model.add(Embedding(max_features, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[9]:


model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


# In[10]:


history=model.fit(X_train, y_train, batch_size=64, epochs=30, validation_data=(X_test, y_test))


# In[11]:


import matplotlib.pyplot as plt

# Assuming that 'history' is the output of the fit method
# history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))

# Plotting loss
plt.figure(figsize=(12,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting accuracy
plt.figure(figsize=(12,6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[17]:


y_pred = model.predict(X_test)


# In[21]:


report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)


# In[25]:


from sklearn.metrics import confusion_matrix


# In[26]:


cm = confusion_matrix(y_test, y_pred)


# In[28]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[30]:


class_labels = np.unique(y_test)


# In[31]:


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:




