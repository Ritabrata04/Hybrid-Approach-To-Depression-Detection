#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("depression_dataset_reddit_cleaned.csv")


# In[3]:


df.head


# In[4]:


df.info()


# In[5]:


class_counts = df['is_depression'].value_counts()


# In[6]:


plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=['Not depressed', 'Depressed'], autopct='%1.1f%%', startangle=90)
plt.title('Class distribution')
plt.show()


# In[8]:


from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk


# In[9]:


nltk.download('stopwords')


# In[10]:


stop_words = set(stopwords.words('english'))


# In[11]:


text = ' '.join(review for review in df['clean_text'])


# In[12]:


text = ' '.join([word for word in text.split() if word not in stop_words])


# In[13]:


wordcloud = WordCloud(background_color="white").generate(text)


# In[14]:


plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[15]:


from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk


# In[16]:


nltk.download('punkt')


# In[17]:


ps = PorterStemmer()


# In[18]:


def stem_sentences(sentence):
    tokens = word_tokenize(sentence)
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


# In[19]:


df['stemmed_text'] = df['clean_text'].apply(stem_sentences)


# In[20]:


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# In[21]:


nltk.download('wordnet')


# In[22]:


lemmatizer = WordNetLemmatizer()


# In[23]:


def lemmatize_sentences(sentence):
    tokens = word_tokenize(sentence)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)


# In[24]:


df['lemmatized_text'] = df['clean_text'].apply(lemmatize_sentences)


# In[25]:


sentence_lengths = df['clean_text'].apply(lambda x: len(x.split()))


# In[29]:


plt.figure(figsize=(8, 6))
plt.hist(sentence_lengths, bins=40, color='skyblue')
plt.title('Histogram of sentence lengths')
plt.xlabel('Sentence length')
plt.ylabel('Frequency')
plt.show()


# In[30]:


from collections import Counter
import numpy as np


# In[31]:


not_depressed_text = ' '.join(df[df['is_depression'] == 0]['clean_text'])
depressed_text = ' '.join(df[df['is_depression'] == 1]['clean_text'])
not_depressed_words = Counter(not_depressed_text.split())
depressed_words = Counter(depressed_text.split())


# In[32]:


not_depressed_common = not_depressed_words.most_common(10)
depressed_common = depressed_words.most_common(10)


# In[33]:


fig, axs = plt.subplots(2, 1, figsize=(10, 12))
axs[0].bar(*zip(*not_depressed_common), color='skyblue')
axs[0].set_title('Not depressed')
axs[0].set_ylabel('Count')
axs[1].bar(*zip(*depressed_common), color='skyblue')
axs[1].set_title('Depressed')
axs[1].set_xlabel('Word')
axs[1].set_ylabel('Count')
plt.tight_layout()
plt.show()


# In[35]:


import seaborn as sns


# In[38]:


df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))

plt.figure(figsize=(10,6))
sns.boxplot(x='is_depression', y='word_count', data=df)
plt.title('Boxplot of Sentence Word Counts by Depression State')
plt.show()

