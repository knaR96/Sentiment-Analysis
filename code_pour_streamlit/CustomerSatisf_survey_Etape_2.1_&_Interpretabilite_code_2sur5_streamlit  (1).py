
# In[8]:


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re

from preprocessed_class_tfidf import comment_cleaner

# In[10]:


#à supprimer

df = pd.read_csv(r"C:\Users\kevin\0_projet_satisfaction_client\reviews_trust.csv")
df = df[["Commentaire", "star"]]
df["Sentiment"] = df["star"].apply(lambda x : np.where(x >=4 , 1 , 0))  # ajout de la colonne Sentiment à df



# In[11]:


#Suppression des valeurs manquantes de la colonne Commentaire
df = df.dropna(axis = 0, how = 'any', subset =["Commentaire"])
df.reset_index(inplace = True)
df = df.drop(['index','star'],axis = 1)


# In[12]:
import nltk
nltk.download('stopwords')


# In[14]:


#Tokenization via RegexpTokenizer

df["preprocessed"] = df["Commentaire"]


#séparation des données pour l'analyse des sentiments
X = df.drop(["Commentaire", "Sentiment"] , axis=1)
y = df["Sentiment"]



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, random_state = 0)


# In[18]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB #Particulièrement adapté aux problèmes de classification avec des features discretes (text classification)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report


# Best score training
from joblib import dump, load
import time

####
#tfidf - LOGREG
start = time.time()
clf_lr_tfidf = LogisticRegression(max_iter= 500, C = 1.0, penalty = 'l2', solver = 'liblinear')
end = time.time()
print("Durée d'entraînement :", end - start, "secondes")

####
#tfidf - Multinomial Naive Bayes

clf_MNB_tfidf = MultinomialNB(alpha=0.5862068965931034)

print("Durée d'entraînement :", end - start, "secondes")

####
#BOW - Multinomial Naive Bayes

clf_MNB_BOW = MultinomialNB(alpha=0.9655172413827586)

print("Durée d'entraînement :", end - start, "secondes")




#### construction pipeline


## fonction de nétoyage du commentaire

from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.feature_extraction.text import TfidfVectorizer
vec_unigram = TfidfVectorizer(analyzer='word', ngram_range=(1,1))
vec_unigram_BOW = CountVectorizer(analyzer='word', ngram_range=(1,1))



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
import joblib
scaler = MaxAbsScaler()

####
# pipe tfidf - LOGREG
pipe_tfidf_lr = Pipeline(steps=[
    ('clean_comment',comment_cleaner()),
    ('vectorizer', vec_unigram),
    ('classifieur',clf_lr_tfidf)], 
    verbose= True)

####
# pipe tfidf - MNB
pipe_tfidf_MNB = Pipeline(steps=[
    ('clean_comment',comment_cleaner()),
    ('vectorizer', vec_unigram),
    ('classifieur',clf_MNB_tfidf)], 
    verbose= True)

####
# pipe tfidf - MNB
pipe_BOW_MNB = Pipeline(steps=[
    ('clean_comment',comment_cleaner()),
    ('vectorizer', vec_unigram_BOW),
    ('scaler', scaler),
    ('classifieur',clf_MNB_tfidf)], 
    verbose= True)

# convert df to array 

X_train = np.array(X_train)

# pipeline training

pipe_tfidf_lr.fit(X_train.reshape(-1,1),y_train)
pipe_tfidf_MNB.fit(X_train.reshape(-1,1),y_train)
pipe_BOW_MNB.fit(X_train.reshape(-1,1),y_train)

X_test = np.array(X_test)

pipe_tfidf_lr.predict(X_test)
pipe_tfidf_MNB.predict(X_test)
pipe_BOW_MNB.predict(X_test)

# Save to file in the current working directory
joblib_file = "C:/Users/kevin/0_projet_satisfaction_client/10_Streamlit/code_pour_streamlit/joblib_file/pipe_tfidf_lr_.pkl"
joblib.dump(pipe_tfidf_lr, joblib_file)

joblib_file = "C:/Users/kevin/0_projet_satisfaction_client/10_Streamlit/code_pour_streamlit/joblib_file/pipe_tfidf_MNB_.pkl"
joblib.dump(pipe_tfidf_MNB, joblib_file)

joblib_file = "C:/Users/kevin/0_projet_satisfaction_client/10_Streamlit/code_pour_streamlit/joblib_file/pipe_BOW_MNB.pkl"
joblib.dump(pipe_BOW_MNB, joblib_file)

print(pipe_tfidf_lr.score(X_test, y_test))
print(pipe_tfidf_MNB.score(X_test, y_test))
print(pipe_BOW_MNB.score(X_test, y_test))

"""
pipe_tfidf_lr = joblib.load(r"C:/Users/kevin/0_projet_satisfaction_client/10_Streamlit/code_pour_streamlit/joblib_file/pipe_tfidf_lr_.pkl")
print(pipe_tfidf_lr)"""



