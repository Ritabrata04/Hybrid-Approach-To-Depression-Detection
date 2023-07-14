#!/usr/bin/env python
# coding: utf-8

# # BERT FOR DEPRESSION DETECTION

# This project aims to utilize a pre-trained model, specifically the BERT (Bidirectional Encoder Representations from Transformers) model, to detect depression from text. The details of the implementation are as follows:

# ## Dataset and Preprocessing

# The dataset used for this task contains two columns - the text data that is labeled as 'clean_text', and a binary label 'is_depression' that indicates whether the text is indicative of depression (1) or not (0).
# 
# The preprocessing phase includes tokenizing the text data into a format that BERT can understand. This includes:
# 
# Tokenization: The sentences are tokenized into words while preserving the context of those words.
# 
# Special Tokens: BERT requires special tokens to distinguish real tokens from padding tokens and also to separate different sentences. Hence '[CLS]' token is added at the beginning of each sentence and '[SEP]' at the end.
# 
# Padding: To maintain a consistent input size, sentences that are shorter than a defined maximum sequence length are padded with '[PAD]' tokens.
# 
# Attention Masks: An attention mask is an array of 1s and 0s indicating which tokens are padding and which aren't to aid in training.

# ## BERT Model and Training

# The BERT model used is 'bert-base-uncased'. The architecture of the BERT model is based on the Transformer architecture, specifically designed to understand the context of a word based on all of its surroundings (left and right of the word). It is a pre-trained model on a large corpus of uncased English text and hence contains rich semantic understanding of the English language, which can be leveraged for this task.
# 
# The BERT model is connected to a dropout layer for regularization and then a dense layer for the binary classification task. The model is trained using the 'BertAdam' optimizer which is a version of the Adam optimizer designed for the BERT model.
# 
# The loss function used is 'binary_crossentropy', which is suitable for a binary classification problem. The metrics used to evaluate the performance of the model during training and validation are 'accuracy', 'Precision', and 'Recall'.

# ## Evaluation

# The performance of the model is evaluated using a variety of metrics. These include accuracy, precision, recall, F1-score, and area under the Receiver Operating Characteristic curve (AUC-ROC). The confusion matrix is also computed to observe the number of true positives, true negatives, false positives, and false negatives.

# In[29]:


import pandas as pd
import numpy as np


# In[30]:


df=pd.read_csv("depression_dataset_reddit_cleaned.csv")


# In[31]:


tweets = df.values[:,0]
labels = df.values[:,1].astype(float)
print (tweets[40], labels[40])


# In[32]:


get_ipython().system('pip install sentence-transformers')


# In[33]:


from sentence_transformers import SentenceTransformer
bert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')


# In[34]:


embeddings = bert_model.encode(tweets, show_progress_bar=True)
print (embeddings.shape)


# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, 
                                          test_size=0.2, random_state=42)
print ("Training set shapes:", X_train.shape, y_train.shape)
print ("Test set shapes:", X_test.shape, y_test.shape)


# In[36]:


from tensorflow.keras import Sequential, layers

classifier = Sequential()
classifier.add (layers.Dense(256, activation='relu', input_shape=(768,)))
classifier.add (layers.Dense(1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
    
hist = classifier.fit (X_train, y_train, epochs=100, batch_size=16, 
                      validation_data=(X_test, y_test))


# In[37]:


from matplotlib import pyplot

pyplot.figure(figsize=(15,5))
pyplot.subplot(1, 2, 1)
pyplot.plot(hist.history['loss'], 'r', label='Training loss')
pyplot.plot(hist.history['val_loss'], 'g', label='Validation loss')
pyplot.legend()
pyplot.subplot(1, 2, 2)
pyplot.plot(hist.history['accuracy'], 'r', label='Training accuracy')
pyplot.plot(hist.history['val_accuracy'], 'g', label='Validation accuracy')
pyplot.legend()
pyplot.show()


# In[38]:


from tensorflow.keras import Sequential, layers

classifier = Sequential()
classifier.add (layers.Dense(256, activation='relu', input_shape=(768,)))
classifier.add (layers.Dense(1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
    
hist = classifier.fit (X_train, y_train, epochs=30, batch_size=16, 
                      validation_data=(X_test, y_test))


# In[39]:


from matplotlib import pyplot

pyplot.figure(figsize=(15,5))
pyplot.subplot(1, 2, 1)
pyplot.plot(hist.history['loss'], 'r', label='Training loss')
pyplot.plot(hist.history['val_loss'], 'g', label='Validation loss')
pyplot.legend()
pyplot.subplot(1, 2, 2)
pyplot.plot(hist.history['accuracy'], 'r', label='Training accuracy')
pyplot.plot(hist.history['val_accuracy'], 'g', label='Validation accuracy')
pyplot.legend()
pyplot.show()


# In[40]:


y_pred = classifier.predict(X_test)


# In[41]:


y_pred = (y_pred > 0.5).astype(int)


# In[42]:


from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)


# In[28]:


from sklearn.metrics import confusion_matrix


# In[43]:


cm = confusion_matrix(y_test, y_pred)


# In[45]:


class_labels = np.unique(y_test)


# In[50]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[51]:


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:




