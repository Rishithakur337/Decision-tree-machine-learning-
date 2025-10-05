#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_iris


# In[3]:


iris=load_iris()


# In[4]:


iris


# In[5]:


iris.target


# In[6]:


import seaborn as sns


# In[7]:


df=sns.load_dataset('iris')


# In[8]:


df.head()


# In[9]:


#independent features and dependent features
x=df.iloc[:,:-1] # sare rows and sare column except last column
y=iris.target


# In[10]:


x


# In[11]:


y


# In[12]:


###train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state = 42)


# In[13]:


x_train


# In[14]:


from sklearn.tree import DecisionTreeClassifier


# In[15]:


#post prunning
treemodel = DecisionTreeClassifier(ccp_alpha=0.01)


# In[16]:


treemodel.fit(x_train,y_train)


# In[17]:


from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(treemodel,filled=True)


# In[18]:


#prediction
y_pred=treemodel.predict(x_test)


# In[19]:


y_pred


# In[20]:


from sklearn.metrics import accuracy_score,classification_report


# In[21]:


score=accuracy_score(y_pred,y_test)
print(score)


# In[22]:


print(classification_report(y_pred,y_test))


# In[23]:


## Preprunning
parameter={
 'criterion':['gini','entropy','log_loss'],
  'splitter':['best','random'],
  'max_depth':[1,2,3,4,5],
  'max_features':['auto', 'sqrt', 'log2']
    
}


# In[25]:


from sklearn.model_selection import GridSearchCV


# In[27]:


treemodel=DecisionTreeClassifier()
cv=GridSearchCV(treemodel,param_grid=parameter,cv=5,scoring='accuracy')


# In[28]:


cv.fit(x_train,y_train)


# In[29]:


cv.best_params_


# In[40]:


y_test


# In[42]:


y_pred=cv.predict(x_test)


# In[43]:


from sklearn.metrics import accuracy_score,classification_report


# In[44]:


score=accuracy_score(y_pred,y_test)


# In[45]:


score


# In[46]:


print(classification_report(y_pred,y_test))

