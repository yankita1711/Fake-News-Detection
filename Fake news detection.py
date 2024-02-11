#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# In[4]:


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier


# In[5]:


df=pd.read_csv('data.csv')


# In[6]:


df.info()


# In[8]:


df.head()


# In[9]:


df.columns


# In[10]:


df.shape


# In[11]:


df.describe()


# In[12]:


from numpy import nan


# In[13]:


df.isna().any()


# In[14]:


df.isnull().sum()


# In[15]:


df=df.drop(['URLs'],axis=1)
df=df.dropna()


# In[20]:


y=df.Label
X=df.Body


# In[21]:


#train_ test separation
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[22]:


#Applying tfidf to the data set
tfidf_vect = TfidfVectorizer(stop_words = 'english')
tfidf_train = tfidf_vect.fit_transform(X_train)
tfidf_test = tfidf_vect.transform(X_test)
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vect.get_feature_names())


# In[23]:


#Applying Naive Bayes
clf = MultinomialNB() 
clf.fit(tfidf_train, y_train)   


# In[24]:


pred = clf.predict(tfidf_test)    


# In[25]:


cm = metrics.confusion_matrix(y_test, pred)
print(cm)


# In[26]:


score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)


# In[ ]:




