#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# In[2]:


df=pd.read_csv("depression_dataset_reddit_cleaned.csv")
df.head()


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:



df["is_depression"].value_counts(normalize=True).plot(kind="pie", autopct="%1.1f%%", colors=["skyblue", "lightgreen"])
plt.axis('equal')
plt.title('Distribution of Depression')
plt.legend(labels=["Non-Depressed", "Depressed"])
plt.show()


# In[5]:


df.info()


# In[6]:


import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re


# In[7]:


nltk.download("all")


# In[8]:


from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential


# In[9]:


w=WordNetLemmatizer()
for i in range(len(df)):
  review=re.sub('[^a-zA-Z]', ' ', df["clean_text"][i])
  review=review.lower()
  review=review.split()
  review=[w.lemmatize(word) for word in review if not word in set(stopwords.words("english"))]
  review=" ".join(review)
  df["clean_text"][i]=review
df.head()


# In[10]:


s=set()
for i in range(len(df)):
    k=df["clean_text"][i].split()
    for j in range(len(k)):
        s.add(k[j])
len(s)


# In[11]:


voc_size=18611
onehot_repr1=[one_hot(words,voc_size)for words in df["clean_text"]]


# In[12]:


max=0
for i in onehot_repr1:
    if len(i)>max:
        max=len(i)
max


# In[13]:


sent_length=max
embedded_docs1=pad_sequences(onehot_repr1,padding='pre',maxlen=sent_length)


# In[14]:


embedding_vector_features=sent_length*2
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add((LSTM(100)))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


# In[15]:


Y=df["is_depression"]


# In[16]:


Y.shape


# In[17]:


embedded_docs1.shape


# In[18]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(embedded_docs1,Y,test_size=0.2,random_state=10,stratify=Y)


# In[20]:


print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)


# In[21]:


model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=40,batch_size=16)


# In[22]:


Y_pred=model.predict(X_test)


# In[23]:


Y_pred=(Y_pred>=0.5).astype("int")


# In[33]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[25]:


cm = confusion_matrix(Y_test, Y_pred)


# In[26]:


class_labels = np.unique(Y_test)


# In[27]:


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# In[28]:


report = classification_report(Y_test, Y_pred)
print("Classification Report:")
print(report)


# In[15]:


# Accuracy values
class0_acc = 0.95
class1_acc = 0.95
overall_acc = 0.95

# Create a horizontal bar chart
fig, ax = plt.subplots(figsize=(8, 4))

# Plot the bars
ax.barh([0, 1, 2], [class0_acc, class1_acc, overall_acc], color=['blue', 'green', 'orange'])

# Add labels and title
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['Class 0 Accuracy', 'Class 1 Accuracy', 'Overall Accuracy'])
ax.set_xlabel('Accuracy')
ax.set_title('Accuracy Visualization')

# Add values next to the bars
for i, v in enumerate([class0_acc, class1_acc, overall_acc]):
    ax.text(v + 0.01, i, str(round(v, 2)), color='black', va='center')

# Display the chart
plt.tight_layout()
plt.show()


# In[ ]:


A different approach


# 

# In[19]:


from tensorflow.keras.layers import Embedding, GRU, Dense


# In[20]:


X = df['clean_text'].values
y = df['is_depression'].values


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

vocab_size = len(tokenizer.word_index) + 1
max_length = 100


# In[23]:


X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


# In[24]:


X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post')


# In[25]:


model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(GRU(128))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[38]:


model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))


# In[39]:


Y_pred = model.predict(X_test)


# In[40]:


Y_pred=(Y_pred>=0.5).astype("int")


# In[41]:


cm = confusion_matrix(Y_test, Y_pred)


# In[42]:


class_labels = np.unique(Y_test)


# In[43]:


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# In[44]:


report = classification_report(Y_test, Y_pred)
print("Classification Report:")
print(report)


# In[ ]:




