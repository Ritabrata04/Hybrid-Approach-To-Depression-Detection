#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# In[1]:


import pandas as pd
df=pd.read_csv("depression_dataset_reddit_cleaned.csv")


# In[9]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])
word_index = tokenizer.word_index


# In[10]:


data = pad_sequences(sequences)


# In[11]:


labels = df['is_depression'].values


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# In[13]:


maxlen = data.shape[1] 
embedding_dim = 100 


# In[14]:


input_seq = Input(shape=(maxlen,))
embedded_seq = Embedding(len(word_index) + 1, embedding_dim, input_length=maxlen)(input_seq)
encoded = LSTM(128, return_sequences=True)(embedded_seq)
encoded = LSTM(64)(encoded)

decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(maxlen, activation='sigmoid')(decoded)

sequence_autoencoder = Model(input_seq, decoded)


# In[15]:


encoder = Model(input_seq, encoded)

sequence_autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[16]:


hist=sequence_autoencoder.fit(X_train, X_train, 
                         epochs=10,  
                         batch_size=64, 
                         shuffle=True,
                         validation_data=(X_test, X_test))


# In[17]:


encoded_text = encoder.predict(data)


# In[18]:


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# In[19]:


tsne_model = TSNE(n_components=2, random_state=0)
tsne_data = tsne_model.fit_transform(encoded_text)


# In[20]:


tsne_df = pd.DataFrame(data = tsne_data, columns = ['Dim_1', 'Dim_2'])
tsne_df = pd.concat([tsne_df, df['is_depression'].reset_index(drop=True)], axis=1)


# In[21]:


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Dim_1', fontsize = 15)
ax.set_ylabel('Dim_2', fontsize = 15)
ax.set_title('2D plot of encoded text', fontsize = 20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets, colors):
    indicesToKeep = tsne_df['is_depression'] == target
    ax.scatter(tsne_df.loc[indicesToKeep, 'Dim_1'], tsne_df.loc[indicesToKeep, 'Dim_2'], c = color, s = 50)
ax.legend(targets)
ax.grid()


# In[22]:


from sklearn.cluster import KMeans


# In[23]:


n_clusters = 2


# In[24]:


kmeans = KMeans(n_clusters=n_clusters)


# In[26]:


kmeans.fit(tsne_df)


# In[27]:


cluster_assignments = kmeans.labels_


# In[32]:


from sklearn.metrics import silhouette_score

score = silhouette_score(tsne_df, cluster_assignments)

print('Silhouette Score: ', score)


# In[36]:


plt.scatter(tsne_df.iloc[:, 0], tsne_df.iloc[:, 1], c=cluster_assignments, cmap='viridis')
plt.title("t-SNE visualization with KMeans clustering")
plt.show()


# In[ ]:




