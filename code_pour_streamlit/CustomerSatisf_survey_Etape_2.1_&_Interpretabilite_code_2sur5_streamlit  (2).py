#!/usr/bin/env python
# coding: utf-8
Ce notebook traite de la première modélisation des sentiments
# In[1]:


pip install git+https://github.com/oracle/Skater.git


# In[2]:


#pip install shap


# In[3]:


# pip install numba.core


# ## Exploration des données (vision d’ensemble sur les données) + Visualisation

# In[8]:


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re


# In[9]:


# importation du DATASET 
#from google.colab import files

#uploaded = files.upload()


# In[10]:


#à supprimer

df = pd.read_csv(r"C:\Users\kevin\0_projet_satisfaction_client\reviews_trust.csv")
df = df[["Commentaire", "star"]]
df["Sentiment"] = df["star"].apply(lambda x : np.where(x >=4 , 1 , 0))  # ajout de la colonne Sentiment à df
sns.countplot(df.Sentiment);
df["Sentiment"].value_counts(normalize = True) # léger déséquilibre des classes mais peut être traité comme un problème non déséquilibré


# In[11]:


#Suppression des valeurs manquantes de la colonne Commentaire
df = df.dropna(axis = 0, how = 'any', subset =["Commentaire"])
df.reset_index(inplace = True)
df = df.drop(['index','star'],axis = 1)


# In[12]:


#mettre les titres comme dans le rapport


# ## Modélisation - itération 1

# ## Prédire la satisfaction d’un client
# Problème de classification (prédire le nombre d'étoiles).
# 

# ## Text mining

# In[13]:


import nltk
nltk.download('stopwords')


# In[14]:


#Tokenization via RegexpTokenizer

from nltk.tokenize.regexp import RegexpTokenizer
tokenizer = RegexpTokenizer("[a-zA-Zéèçê]{2,}|[!?.]")
df["preprocessed"] = df["Commentaire"].apply(lambda x : " ".join(x for x in str(x).strip().split()))

# On remplace les éèê par e
def remplacement_carac_e(com):
    text=re.sub(r"[éèêë]","e",com)
    return text
df["preprocessed"] = df["preprocessed"].apply(lambda x : remplacement_carac_e(x))

def neg_identify(text): 
    text = tokenizer.tokenize(text)
    for pos, word in enumerate(text) :
        if pos != len(text)-1 :
            if (word == 'ne' or word == "n" or word =="n'"):
                text[pos+1] = "NON_"+text[pos+1]
    text = " ".join(text)
    return (text)
df["preprocessed"] = df["preprocessed"].apply(lambda x : neg_identify(x))

# Déclaration des stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('french'))
stop_words.update (["_",":",",",";","-","--","...","'","...","'",',',',','…la','la','le','les','..','…','(',')','a+','+','etc…','qq','``',"j'","j '"])
# après une première visualisation des données, "commande" est très freqement apparu dans les 2 catégories et n'apporte à priori pas d'information sur la satisfaction du client"
# print(stop_words)

# Definition d'une fonction de filtrage de stopwords
def stopwords_filtering(chaine): # fonction renvoyant une liste ne contenant pas les stopwords
    tokens =[]
    chaine = tokenizer.tokenize(chaine)
    for mot in chaine :
        if mot not in stop_words :#conservation des mots non stopwords 
            tokens.append(mot)
    tokens = " ".join(tokens)
    return tokens
df["preprocessed"] = df["preprocessed"].apply(lambda x : stopwords_filtering(x))

# Opération de stemming

from nltk.stem.snowball import FrenchStemmer

stemmer = FrenchStemmer()
def  stemming(text) :
    text = tokenizer.tokenize(text)
    stemmed_text = ""
    for mot in text: 
        stem = stemmer.stem(mot)
        stemmed_text =  stemmed_text +" "+ stem
    return  stemmed_text

df["preprocessed"] = df["preprocessed"].apply(lambda x: stemming(x))


# In[15]:


df


# In[16]:


#séparation des données pour l'analyse des sentiments
X = df.drop(["Commentaire", "Sentiment"] , axis=1)
y = df["Sentiment"]


# ## Analyse de sentiments 

# ### Méthode BOW (Bag of words) CountVectorizer

# In[17]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 

df_train, df_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, random_state = 0)


# In[18]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB #Particulièrement adapté aux problèmes de classification avec des features discretes (text classification)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report


# In[12]:


# ##  Algorithmes de classification sans tunning d'hyperparamètres
# - Régression logistique
# - SVM
# - Naïve Bayes
# - GradientBoosting
# Vectorisation prenant en compte uniquement les unigrams (vectorisation mot / mot)


# In[12]:


get_ipython().run_cell_magic('time', '', 'from scipy.sparse import hstack\nimport time \n# Vectorization avec CountVectorizer ()\nfrom sklearn.preprocessing import MaxAbsScaler\nscaler = MaxAbsScaler()\n\nvec_unigram = CountVectorizer(analyzer=\'word\', ngram_range=(1,1)) #stopwords déjà supprimés dans le prétraitement\nX_train_text = vec_unigram.fit_transform(df_train.preprocessed)\nX_test_text = vec_unigram.transform(df_test.preprocessed)\n#on ajoute les metadonnées standardisées à notre vecteur d\'occurence\n\n\nX_train = hstack([X_train_text,df_train.drop("preprocessed", axis=1).values])\nX_test = hstack([X_test_text,df_test.drop("preprocessed", axis=1).values])\n\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)\n\n# instanciation des classifieurs\n\nclf_reglog_unigram = LogisticRegression(C=1, max_iter= 10000)\n\nclf_svc_unigram = SVC(probability=True)\n\nclf_MNB_unigram = MultinomialNB() \n\nclf_GB_unigram = GradientBoostingClassifier()\n\n# Fit des classifieurs aux données d\'entraînement\nstart = time.time()\nclf_reglog_unigram.fit(X_train, y_train)\nend = time.time()\nprint("The time of clf_reglog_unigram:", end-start)\n# start = time.time()\n# clf_svc_unigram.fit(X_train, y_train)\n# end = time.time()\n# print("The time of clf_svc_unigram:", end-start)\n# start = time.time()\nclf_MNB_unigram.fit(X_train, y_train)\nend = time.time()\nprint("The time of clf_MNB_unigram:", end-start)\nstart = time.time()\nclf_GB_unigram.fit(X_train, y_train)\nend = time.time()\nprint("The time of clf_GB_unigram:", end-start)')


# In[13]:


get_ipython().run_cell_magic('time', '', '# Affichage des scores des différents classifieurs\n\nprint(" -- Régression logistique --")\nprint("Score sur le trainset :",clf_reglog_unigram.score(X_train,y_train),"; Score sur le testset : ",clf_reglog_unigram.score(X_test,y_test))\n# print(" -- SVC --")\n# print("Score sur le trainset :",clf_svc_unigram.score(X_train,y_train),"; Score sur le testset : ",clf_svc_unigram.score(X_test,y_test))\n# print(" -- Multinomial Naïve Bayes --")\n# print("Score sur le trainset :",clf_MNB_unigram.score(X_train,y_train),"; Score sur le testset : ",clf_MNB_unigram.score(X_test,y_test))\nprint(" -- GradientBoosting -- ")\nprint("Score sur le trainset :",clf_GB_unigram.score(X_train,y_train),"; Score sur le testset : ",clf_GB_unigram.score(X_test,y_test))')


# In[14]:


get_ipython().run_cell_magic('time', '', '# Affichage des prédictions par les différents classifieurs\nprint(" -- Régression logistique --")\npred_clf_reglog_unigram = clf_reglog_unigram.predict(X_test)\ndisplay(pd.crosstab(y_test, pred_clf_reglog_unigram,  colnames=["Classe réelle"], rownames=["Classe prédite"]))\nprint(classification_report(y_test,pred_clf_reglog_unigram))')


# ## Interprétabilité de régression logistique unigram

# In[15]:


import shap
features =  vec_unigram.get_feature_names() + ['CAPSLOCK', 'exclamation','interogation','chainpoints','nb_caracter']
explainer = shap.LinearExplainer(clf_reglog_unigram,X_train)
x = X_test.toarray()
x_test = pd.DataFrame(x, columns=features)
shap_values = explainer(x_test)
figure = plt.figure(figsize=(10,10))
shap.plots.beeswarm(shap_values,max_display=25)


# In[20]:


# print(" -- SVC --")
# pred_clf_svc_unigram = clf_svc_unigram.predict(X_test)
# display(pd.crosstab(y_test, pred_clf_svc_unigram,  colnames=["Classe réelle"], rownames=["Classe prédite"]))
# print(classification_report(y_test,pred_clf_svc_unigram))


# ### Interpretabilite de SVC
# 

# In[21]:


from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

# def svc_predict_X(x):
#     return clf_svc_unigram.predict_proba(x)


# In[22]:


# interpreter = Interpretation(X_train.toarray(), feature_names=features)
# model = InMemoryModel(svc_predict_X, examples = X_train.toarray())


# In[23]:


#interpreter.feature_importance.plot_feature_importance(model, ascending=True)
#cela prend à peu prés 4 jours pour l'exécuter 
#si on essaie de prendre que des échantillions, l'interprétabilité ne va pas être précise.


# In[24]:


# print(" -- Multinomial Naïve Bayes --")
# pred_clf_MNB_unigram = clf_MNB_unigram.predict(X_test)
# display(pd.crosstab(y_test, pred_clf_MNB_unigram,  colnames=["Classe réelle"], rownames=["Classe prédite"]))
# print(classification_report(y_test,pred_clf_MNB_unigram))
print("l'exécution dee l'interprétation de ces deux modèles est trés longue")


# ### Interpretabilite de Multinomial naive Bayes
# 

# In[25]:


# from skater.core.explanations import Interpretation
# from skater.model import InMemoryModel

# def predict_X(x):
#     return clf_MNB_unigram.predict_proba(x)


# In[26]:


# interpreter = Interpretation(X_train.toarray(), feature_names=features)
# model = InMemoryModel(predict_X, examples = X_train.toarray())


# In[27]:


#interpreter.feature_importance.plot_feature_importance(model, ascending=True)
#ça prend bcp de temps pour l'excéuter
#si on essaie de prendre que des échantillions, l'interprétabilité ne sera précise.


# In[28]:


plt.figure(figsize=(14,5))
neg_class_prob_sorted = clf_MNB_unigram.feature_log_prob_[0, :].argsort()[::-1]
pos_class_prob_sorted = clf_MNB_unigram.feature_log_prob_[1, :].argsort()[::1]

feauture_amount = 20
neg_class_prob_feature = np.take(features, neg_class_prob_sorted[:feauture_amount])
pos_class_prob_feature = np.take(features, pos_class_prob_sorted[:feauture_amount])
plt.subplot(121)
plt.barh(neg_class_prob_feature, sorted(list(neg_class_prob_sorted))[:feauture_amount])
plt.title("features importance des mots négatifs");
plt.subplot(122)
plt.barh(pos_class_prob_feature, sorted(list(pos_class_prob_sorted))[:feauture_amount])
plt.title("features importance des mots positifs");


# In[29]:


#list(neg_class_prob_sorted).plot.barh()


# In[30]:


print(np.take(features, neg_class_prob_sorted[:10]))
print(np.take(features, pos_class_prob_sorted[:10]))


# ## Interprétabilité GBS

# In[31]:


print(" -- GradientBoosting -- ")
pred_clf_GB_unigram = clf_GB_unigram.predict(X_test)
display(pd.crosstab(y_test, pred_clf_GB_unigram,  colnames=["Classe réelle"], rownames=["Classe prédite"]))
print(classification_report(y_test,pred_clf_GB_unigram))


# In[32]:


import shap
explainer = shap.TreeExplainer(clf_GB_unigram)
x = X_test.toarray()
x_test = pd.DataFrame(x, columns=features)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values, x_test, max_display=25)


# In[33]:


# ##  Algorithmes de classification sans tunning d'hyperparamètres
# - Régression logistique
# - SVM
# - Naïve Bayes
# - GradientBoosting
# Vectorisation prenant en compte des ngrams (1 à 2)


# In[34]:


get_ipython().run_cell_magic('time', '', 'import time \n# Vectorization avec CountVectorizer ()\n\ndf_train, df_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, random_state = 0)\n\nvec_ngram = CountVectorizer(analyzer=\'word\', ngram_range=(1,2)) #stopwords déjà supprimés dans le prétraitement\nX_train_text = vec_ngram.fit_transform(df_train.preprocessed)\nX_test_text = vec_ngram.transform(df_test.preprocessed)\n#on ajoute les metadonnées standardisées à notre vecteur d\'occurence\n\nX_train = hstack([X_train_text,df_train.drop("preprocessed", axis=1).values])\nX_test = hstack([X_test_text,df_test.drop("preprocessed", axis=1).values])\n\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)\n\n\n# instanciation des classifieurs \nclf_reglog_ngram = LogisticRegression(C=1, max_iter= 10000)\nclf_svc_ngram = SVC()\nclf_MNB_ngram = MultinomialNB() \nclf_GB_ngram = GradientBoostingClassifier() \n\n# Fit des classifieurs aux données d\'entraînement\nstart = time.time()\nclf_reglog_ngram.fit(X_train, y_train)\nend = time.time()\nprint("The time of clf_reglog_ngram:", end-start)\nstart = time.time()\nclf_svc_ngram.fit(X_train, y_train)\nend = time.time()\nprint("The time of clf_svc_ngram:", end-start)\nstart = time.time()\nclf_MNB_ngram.fit(X_train, y_train)\nend = time.time()\nprint("The time of clf_MNB_ngram:", end-start)\nstart = time.time()\nclf_GB_ngram.fit(X_train, y_train)\nend = time.time()\nprint("The time of clf_GB_ngram:", end-start)\n')


# In[35]:


get_ipython().run_cell_magic('time', '', '# Affichage des scores des différents classifieurs\nprint(" -- Régression logistique --")\nprint("Score sur le trainset :",clf_reglog_ngram.score(X_train,y_train),"; Score sur le testset : ",clf_reglog_ngram.score(X_test,y_test))\nprint(" -- SVC --")\nprint("Score sur le trainset :",clf_svc_ngram.score(X_train,y_train),"; Score sur le testset : ",clf_svc_ngram.score(X_test,y_test))\nprint(" -- Multinomial Naïve Bayes --")\nprint("Score sur le trainset :",clf_MNB_ngram.score(X_train,y_train),"; Score sur le testset : ",clf_MNB_ngram.score(X_test,y_test))\nprint(" -- GradientBoosting -- ")\nprint("Score sur le trainset :",clf_GB_ngram.score(X_train,y_train),"; Score sur le testset : ",clf_GB_ngram.score(X_test,y_test))')


# In[36]:


get_ipython().run_cell_magic('time', '', '# Affichage des prédictions par les différents classifieurs\nprint(" -- Régression logistique --")\npred_clf_reglog_ngram = clf_reglog_ngram.predict(X_test)\ndisplay(pd.crosstab(y_test, pred_clf_reglog_ngram,  colnames=["Classe réelle"], rownames=["Classe prédite"]))\nprint(classification_report(y_test,pred_clf_reglog_ngram))\n#print(" -- SVC --")\npred_clf_svc_ngram = clf_svc_ngram.predict(X_test)\ndisplay(pd.crosstab(y_test, pred_clf_svc_ngram,  colnames=["Classe réelle"], rownames=["Classe prédite"]))\nprint(classification_report(y_test,pred_clf_svc_ngram))\nprint(" -- Multinomial Naïve Bayes --")\npred_clf_MNB_ngram = clf_MNB_ngram.predict(X_test)\ndisplay(pd.crosstab(y_test, pred_clf_MNB_ngram,  colnames=["Classe réelle"], rownames=["Classe prédite"]))\nprint(classification_report(y_test,pred_clf_MNB_ngram))\nprint(" -- GradientBoosting -- ")\npred_clf_GB_ngram = clf_GB_ngram.predict(X_test)\ndisplay(pd.crosstab(y_test, pred_clf_GB_ngram,  colnames=["Classe réelle"], rownames=["Classe prédite"]))\nprint(classification_report(y_test,pred_clf_GB_ngram))')


# In[37]:


### Mise au point du modèle Multinomial Naïve Bayes (prenant en compte uniquement des unigrams)


# In[15]:


get_ipython().run_cell_magic('time', '', 'from scipy.sparse import hstack\nimport time \n# Vectorization avec CountVectorizer ()\nfrom sklearn.preprocessing import MaxAbsScaler\nscaler = MaxAbsScaler()\n\nvec_unigram = CountVectorizer(analyzer=\'word\', ngram_range=(1,1)) #stopwords déjà supprimés dans le prétraitement\nX_train_text = vec_unigram.fit_transform(df_train.preprocessed)\nX_test_text = vec_unigram.transform(df_test.preprocessed)\n#on ajoute les metadonnées standardisées à notre vecteur d\'occurence\n\nX_train = hstack([X_train_text,df_train.drop("preprocessed", axis=1).values])\nX_test = hstack([X_test_text,df_test.drop("preprocessed", axis=1).values])\n\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)\n')


# ###  GridSearch

# In[16]:



#GridSearchCV va nous permettre de tester une série de paramètres et de comparer 
#les performances pour en déduire le meilleur paramétrage.
from sklearn.model_selection import GridSearchCV
params = {"alpha": np.linspace(1e-10,1,30)}
grid_MN = GridSearchCV(MultinomialNB(), cv= 5, param_grid = params)
grid_MN.fit(X_train, y_train)
print("Meilleur paramètre : ", grid_MN.best_params_, "permettant d'obtenir un score de " , grid_MN.best_score_)
print("Score sur le trainset :",grid_MN.score(X_train,y_train),"; Score sur le testset : ",grid_MN.score(X_test,y_test))
print("On ne fait à priori pas face à un problème d'overfiting")


# In[ ]:


# Best score training


# In[17]:


# Best score training
import time

start = time.time()
clf_MNB_bow = MultinomialNB(alpha=0.9655172413827586)
clf_MNB_bow.fit(X_train,y_train)
end = time.time()

print("Durée d'entraînement :", end - start, "secondes")


# In[18]:


#La prédiction
from sklearn.metrics import classification_report
y_pred_MNB_bow_proba = clf_MNB_bow.predict_proba(X_test)
from sklearn.metrics import auc, roc_curve

fpr , tpr , seuil = roc_curve(y_test, y_pred_MNB_bow_proba[:,0], pos_label= 0)
aucf= auc(fpr, tpr)

plt.plot(fpr, tpr, color='coral', lw=2, label ='auc=%1.5f' % aucf)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("-- Multinomial Naïve Bayes ROC CURVE --")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - specificite (Taux de FN)', fontsize=14)
plt.ylabel('Sensibilite (Taux de VP)', fontsize=14)
plt.legend();
plt.show();

y_pred_MNB_bow = clf_MNB_bow.predict(X_test)
print(classification_report(y_test, y_pred_MNB_bow))


# In[41]:


### Mise au point du modèle Multinomial Naïve Bayes (prenant en compte les  unigrams & bigrams)


# In[19]:


# Vectorization avec CountVectorizer ()

df_train, df_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, random_state = 0)

vec_ngram = CountVectorizer(analyzer='word', ngram_range=(1,2)) #stopwords déjà supprimés dans le prétraitement
X_train_text = vec_ngram.fit_transform(df_train.preprocessed)
X_test_text = vec_ngram.transform(df_test.preprocessed)
#on ajoute les metadonnées standardisées à notre vecteur d'occurence

X_train = hstack([X_train_text,df_train.drop("preprocessed", axis=1).values])
X_test = hstack([X_test_text,df_test.drop("preprocessed", axis=1).values])

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


params = {"alpha": np.linspace(1e-10,1,30)}
grid_MN_ngram = GridSearchCV(MultinomialNB(), cv= 5, param_grid = params)
grid_MN_ngram.fit(X_train, y_train)
print("Meilleur paramètre : ", grid_MN_ngram.best_params_, "permettant d'obtenir un score de " , grid_MN_ngram.best_score_)
print("Score sur le trainset :",grid_MN_ngram.score(X_train,y_train),"; Score sur le testset : ",grid_MN_ngram.score(X_test,y_test))
print("On fait probablement face à un problème d'overfiting")


# In[20]:


y_pred_MNB_ngram = grid_MN_ngram.predict_proba(X_test)
from sklearn.metrics import auc, roc_curve

fpr , tpr , seuil = roc_curve(y_test, y_pred_MNB_ngram[:,1], pos_label= 1)
aucf= auc(fpr, tpr)
plt.plot(fpr, tpr, color='coral', lw=2, label ='auc=%1.5f' % aucf)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("-- Multinomial Naïve Bayes ROC CURVE --")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - specificite (Taux de FN)', fontsize=14)
plt.ylabel('Sensibilite (Taux de VP)', fontsize=14)
plt.legend();
plt.show();


# # Mise au point du modèle Regression logistique (prenant en compte uniquement des unigrams)

# In[21]:


get_ipython().run_cell_magic('time', '', 'from scipy.sparse import hstack\nimport time \n# Vectorization avec CountVectorizer ()\nfrom sklearn.preprocessing import MaxAbsScaler\nscaler = MaxAbsScaler()\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import GridSearchCV\n\nvec_unigram = CountVectorizer(analyzer=\'word\', ngram_range=(1,1)) #stopwords déjà supprimés dans le prétraitement\nX_train_text = vec_unigram.fit_transform(df_train.preprocessed)\nX_test_text = vec_unigram.transform(df_test.preprocessed)\n#on ajoute les metadonnées standardisées à notre vecteur d\'occurence\n\nX_train = hstack([X_train_text,df_train.drop("preprocessed", axis=1).values])\nX_test = hstack([X_test_text,df_test.drop("preprocessed", axis=1).values])\n\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)\n\n# define models and parameters\nmodel = LogisticRegression(max_iter= 500)\nsolvers = [\'newton-cg\', \'lbfgs\', \'liblinear\']\npenalty = [\'l2\']\nC = [100, 10, 1.0, 0.1, 0.01]\ngrid = dict(solver=solvers,penalty=penalty,C=C)\n# define grid search\ngrid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=5, scoring=\'accuracy\')\ngrid_lr = grid_search.fit(X_train, y_train)\n# summarize results\nprint("Best: %f using %s" % (grid_lr.best_score_, grid_lr.best_params_))\nmeans = grid_lr.cv_results_[\'mean_test_score\']\nstds = grid_lr.cv_results_[\'std_test_score\']\nparams = grid_lr.cv_results_[\'params\']\nfor mean, stdev, param in zip(means, stds, params):\n    print("%f (%f) with: %r" % (mean, stdev, param))')


# In[22]:


# Best score training
import time

start = time.time()
clf_lr_bow = LogisticRegression(max_iter= 500, C = 1, penalty = 'l2', solver = 'newton-cg')
clf_lr_bow.fit(X_train,y_train)
end = time.time()

print("Durée d'entraînement :", end - start, "secondes")


# In[23]:


print("Score sur le trainset :",clf_lr_bow.score(X_train,y_train),"; Score sur le testset : ",clf_lr_bow.score(X_test,y_test))


# In[24]:


y_pred_lr_proba = clf_lr_bow.predict_proba(X_test)

from sklearn.metrics import auc, roc_curve
from sklearn.metrics import classification_report

fpr , tpr , seuil = roc_curve(y_test, y_pred_lr_proba[:,1], pos_label= 1)
aucf= auc(fpr, tpr)

plt.plot(fpr, tpr, color='coral', lw=2, label ='auc=%1.5f' % aucf)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("-- Multinomial Naïve Bayes ROC CURVE --")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - specificite (Taux de FN)', fontsize=14)
plt.ylabel('Sensibilite (Taux de VP)', fontsize=14)
plt.legend();
plt.show();


y_pred_lr = clf_lr_bow.predict(X_test)
print(classification_report(y_test, y_pred_lr))


# In[48]:


### Mise au point du modèle de regression logistique (prenant en compte les  unigrams & bigrams)


# In[49]:


get_ipython().run_cell_magic('time', '', '\n# Vectorization avec CountVectorizer ()\n\ndf_train, df_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, random_state = 0)\n\nvec_ngram = CountVectorizer(analyzer=\'word\', ngram_range=(1,2)) #stopwords déjà supprimés dans le prétraitement\nX_train_text = vec_ngram.fit_transform(df_train.preprocessed)\nX_test_text = vec_ngram.transform(df_test.preprocessed)\n#on ajoute les metadonnées standardisées à notre vecteur d\'occurence\n\nX_train = hstack([X_train_text,df_train.drop("preprocessed", axis=1).values])\nX_test = hstack([X_test_text,df_test.drop("preprocessed", axis=1).values])\n\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)\n\n# define models and parameters\nmodel = LogisticRegression(max_iter= 500)\nsolvers = [\'newton-cg\', \'lbfgs\', \'liblinear\']\npenalty = [\'l2\']\nC = [100, 10, 1.0, 0.1, 0.01]\ngrid = dict(solver=solvers,penalty=penalty,C=C)\n# define grid search\ngrid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=5, scoring=\'accuracy\')\ngrid_lr = grid_search.fit(X_train, y_train)\n# summarize results\nprint("Best: %f using %s" % (grid_lr.best_score_, grid_lr.best_params_))\nmeans = grid_lr.cv_results_[\'mean_test_score\']\nstds = grid_lr.cv_results_[\'std_test_score\']\nparams = grid_lr.cv_results_[\'params\']\nfor mean, stdev, param in zip(means, stds, params):\n    print("%f (%f) with: %r" % (mean, stdev, param))')


# In[50]:


print("Score sur le trainset :",grid_lr.score(X_train,y_train),"; Score sur le testset : ",grid_lr.score(X_test,y_test))


# In[51]:


y_pred_lr = grid_lr.predict_proba(X_test)
from sklearn.metrics import auc, roc_curve

fpr , tpr , seuil = roc_curve(y_test, y_pred_lr[:,1], pos_label= 1)
aucf= auc(fpr, tpr)

plt.plot(fpr, tpr, color='coral', lw=2, label ='auc=%1.5f' % aucf)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("-- Multinomial Naïve Bayes ROC CURVE --")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - specificite (Taux de FN)', fontsize=14)
plt.ylabel('Sensibilite (Taux de VP)', fontsize=14)
plt.legend();
plt.show();


# In[52]:


###  Mise au point du modèle SVC (prenant en compte uniquement des unigrams)


# In[53]:


get_ipython().run_cell_magic('time', '', '\nvec_unigram = CountVectorizer(analyzer=\'word\', ngram_range=(1,1)) #stopwords déjà supprimés dans le prétraitement\nX_train_text = vec_unigram.fit_transform(df_train.preprocessed)\nX_test_text = vec_unigram.transform(df_test.preprocessed)\n#on ajoute les metadonnées standardisées à notre vecteur d\'occurence\n\nX_train = hstack([X_train_text,df_train.drop("preprocessed", axis=1).values])\nX_test = hstack([X_test_text,df_test.drop("preprocessed", axis=1).values])\n\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)\n\n# define models and parameters\nmodel = SVC(probability= True)\nkernel = [\'linear\',\'poly\', \'rbf\', \'sigmoid\']\ngamma = [\'scale\']\nC = [1,10,20,30,40,50]\ngrid = dict(kernel=kernel,C=C,gamma=gamma)\n# define grid search\ngrid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=5, scoring=\'accuracy\')\ngrid_svc = grid_search.fit(X_train, y_train)\n# summarize results\nprint("Best: %f using %s" % (grid_svc.best_score_, grid_svc.best_params_))\nmeans = grid_svc.cv_results_[\'mean_test_score\']\nstds = grid_svc.cv_results_[\'std_test_score\']\nparams = grid_svc.cv_results_[\'params\']\nfor mean, stdev, param in zip(means, stds, params):\n    print("%f (%f) with: %r" % (mean, stdev, param))')


# In[54]:


print("Score sur le trainset :",grid_svc.score(X_train,y_train),"; Score sur le testset : ",grid_svc.score(X_test,y_test))


# In[55]:


y_pred_svc = grid_svc.predict_proba(X_test)
from sklearn.metrics import auc, roc_curve

fpr , tpr , seuil = roc_curve(y_test, y_pred_svc[:,1], pos_label= 1)
aucf= auc(fpr, tpr)

plt.plot(fpr, tpr, color='coral', lw=2, label ='auc=%1.5f' % aucf)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("-- Multinomial Naïve Bayes ROC CURVE --")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - specificite (Taux de FN)', fontsize=14)
plt.ylabel('Sensibilite (Taux de VP)', fontsize=14)
plt.legend();
plt.show();


# In[56]:


### Mise au point du modèle SVC (prenant en compte les  unigrams & bigrams)


# In[57]:


get_ipython().run_cell_magic('time', '', '\n# Vectorization avec CountVectorizer ()\n\ndf_train, df_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, random_state = 0)\n\nvec_ngram = CountVectorizer(analyzer=\'word\', ngram_range=(1,2)) #stopwords déjà supprimés dans le prétraitement\nX_train_text = vec_ngram.fit_transform(df_train.preprocessed)\nX_test_text = vec_ngram.transform(df_test.preprocessed)\n#on ajoute les metadonnées standardisées à notre vecteur d\'occurence\n\nX_train = hstack([X_train_text,df_train.drop("preprocessed", axis=1).values])\nX_test = hstack([X_test_text,df_test.drop("preprocessed", axis=1).values])\n\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)\n\n# define models and parameters\nmodel = SVC(probability = True)\nkernel = [\'linear\',\'rbf\']\ngamma = [\'scale\']\nC = [1,10,20]\ngrid = dict(kernel=kernel,C=C,gamma=gamma)\n# define grid search\ngrid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=5, scoring=\'accuracy\')\ngrid_svc = grid_search.fit(X_train, y_train)\n# summarize results\nprint("Best: %f using %s" % (grid_svc.best_score_, grid_svc.best_params_))\nmeans = grid_svc.cv_results_[\'mean_test_score\']\nstds = grid_svc.cv_results_[\'std_test_score\']\nparams = grid_svc.cv_results_[\'params\']\nfor mean, stdev, param in zip(means, stds, params):\n    print("%f (%f) with: %r" % (mean, stdev, param))')


# In[58]:


print("Score sur le trainset :",grid_svc.score(X_train,y_train),"; Score sur le testset : ",grid_svc.score(X_test,y_test))


# In[59]:


y_pred_svc = grid_svc.predict_proba(X_test)
from sklearn.metrics import auc, roc_curve

fpr , tpr , seuil = roc_curve(y_test, y_pred_svc[:,1], pos_label= 1)
aucf= auc(fpr, tpr)

plt.plot(fpr, tpr, color='coral', lw=2, label ='auc=%1.5f' % aucf)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("-- Multinomial Naïve Bayes ROC CURVE --")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - specificite (Taux de FN)', fontsize=14)
plt.ylabel('Sensibilite (Taux de VP)', fontsize=14)
plt.legend();
plt.show();


# In[60]:


###  Mise au point du modèle GradientBoosting (prenant en compte uniquement des unigrams)


# In[61]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import GridSearchCV\n\ndf_train, df_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, random_state = 0)\n\nvec_unigram = CountVectorizer(analyzer=\'word\', ngram_range=(1,1)) #stopwords déjà supprimés dans le prétraitement\nX_train_text = vec_unigram.fit_transform(df_train.preprocessed)\nX_test_text = vec_unigram.transform(df_test.preprocessed)\n#on ajoute les metadonnées standardisées à notre vecteur d\'occurence\n\nX_train = hstack([X_train_text,df_train.drop("preprocessed", axis=1).values])\nX_test = hstack([X_test_text,df_test.drop("preprocessed", axis=1).values])\n\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)\n\n\n#test hyperparamètres 1\n# params_1 = {\'learning_rate\':[0.01,0.05,0.1,0.13,0.14], \'n_estimators\':[1750,2000,2500,3000]}\n# params_2 = {\'n_estimators\':[3000, 4000,7000]}\n# params_3 = {\'max_depth\':[7,8,10,12] }\n# params_4 = {\'min_samples_split\':[2,4,6,8,10,20,40,60,100], \'min_samples_leaf\':[1,3,5,7,9]}\nparams_5 = {\'max_features\':[2,3,4,5,6,7]}\n\nmodel = GradientBoostingClassifier(learning_rate = 0.14 ,n_estimators = 4000, max_depth = 7, min_samples_leaf=1, min_samples_split = 2, subsample=1, random_state=0)\n\n# define grid search\ngrid_search = GridSearchCV(estimator=model, param_grid=params_5, n_jobs=-1, cv=5, scoring=\'accuracy\')\n#grid_svc = grid_search.fit(X_train, y_train)\n# summarize results\nprint("Best: %f using %s" % (grid_svc.best_score_, grid_svc.best_params_))\nmeans = grid_svc.cv_results_[\'mean_test_score\']\nstds = grid_svc.cv_results_[\'std_test_score\']\nparams = grid_svc.cv_results_[\'params\']\nfor mean, stdev, param in zip(means, stds, params):\n    print("%f (%f) with: %r" % (mean, stdev, param))\nprint(\'Accuracy of the GBM ontrain set: {:.3f}\'.format(grid_svc.score(X_train, y_train)))\nprint(\'Accuracy of the GBM on test set: {:.3f}\'.format(grid_svc.score(X_test, y_test)))\n\nprint("Score sur le trainset :",grid_svc.score(X_train,y_train),"; Score sur le testset : ",grid_svc.score(X_test,y_test))\n\ny_pred_GB=grid_svc.predict(X_test)\n\n\nprint(classification_report(y_test, y_pred_GB))')


# In[62]:


#ajouter l'interprétabilité Gridsearch Gradient boosting ici


# 
# ## Méthode TF-IDF

# In[63]:


#rajouter l'interprétabilité


# In[64]:


# ##  Algorithmes de classification sans tunning d'hyperparamètres
# - Régression logistique
# - SVM
# - Naïve Bayes
# - GradientBoosting
# Vectorisation prenant en compte uniquement les unigrams (vectorisation mot / mot)


# In[21]:


get_ipython().run_cell_magic('time', '', 'from scipy.sparse import hstack\nfrom sklearn.feature_extraction.text import TfidfVectorizer\n\nvec_unigram = TfidfVectorizer(analyzer=\'word\', ngram_range=(1,1)) #stopwords déjà supprimés dans le prétraitement\nX_train_text = vec_unigram.fit_transform(df_train.preprocessed)\nX_test_text = vec_unigram.transform(df_test.preprocessed)\n#on ajoute les metadonnées standardisées à notre vecteur d\'occurence\n\nX_train = hstack([X_train_text,df_train.drop("preprocessed", axis=1).values])\nX_test = hstack([X_test_text,df_test.drop("preprocessed", axis=1).values])\n\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)\n\n\n# # instanciation des classifieurs \n# clf_reglog_unigram = LogisticRegression(C=1, max_iter= 10000)\n# #clf_svc_unigram = SVC()\n# #clf_MNB_unigram = MultinomialNB() \n# clf_GB_unigram = GradientBoostingClassifier()\n\n# # Fit des classifieurs aux données d\'entraînement\n# start = time.time()\n# clf_reglog_unigram.fit(X_train, y_train)\n# end = time.time()\n# print("The time of clf_reglog_unigram:", end-start)\n# start = time.time()\n# clf_svc_unigram.fit(X_train, y_train)\n# end = time.time()\n# #print("The time of clf_svc_unigram:", end-start)\n# start = time.time()\n# clf_MNB_unigram.fit(X_train, y_train)\n# end = time.time()\n# #print("The time of clf_MNB_unigram:", end-start)\n# start = time.time()\n# clf_GB_unigram.fit(X_train, y_train)\n# end = time.time()\n# print("The time of clf_GB_unigram:", end-start)')


# In[66]:


# %%time
# # Affichage des scores des différents classifieurs

# print(" -- Régression logistique --")
# print("Score sur le trainset :",clf_reglog_unigram.score(X_train,y_train),"; Score sur le testset : ",clf_reglog_unigram.score(X_test,y_test))
# #print(" -- SVC --")
# #print("Score sur le trainset :",clf_svc_unigram.score(X_train,y_train),"; Score sur le testset : ",clf_svc_unigram.score(X_test,y_test))
# #print(" -- Multinomial Naïve Bayes --")
# print("Score sur le trainset :",clf_MNB_unigram.score(X_train,y_train),"; Score sur le testset : ",clf_MNB_unigram.score(X_test,y_test))
# print(" -- GradientBoosting -- ")
# print("Score sur le trainset :",clf_GB_unigram.score(X_train,y_train),"; Score sur le testset : ",clf_GB_unigram.score(X_test,y_test))


# In[67]:


get_ipython().run_cell_magic('time', '', '# Affichage des prédictions par les différents classifieurs\nprint(" -- Régression logistique --")\npred_clf_reglog = clf_reglog_unigram.predict(X_test)\ndisplay(pd.crosstab(y_test, pred_clf_reglog,  colnames=["Classe réelle"], rownames=["Classe prédite"]))\nprint(classification_report(y_test,pred_clf_reglog))\nprint(" -- SVC --")\npred_clf_svc = clf_svc_unigram.predict(X_test)\ndisplay(pd.crosstab(y_test, pred_clf_svc,  colnames=["Classe réelle"], rownames=["Classe prédite"]))\nprint(classification_report(y_test,pred_clf_svc))\nprint(" -- Multinomial Naïve Bayes --")\npred_clf_MNB = clf_MNB_unigram.predict(X_test)\ndisplay(pd.crosstab(y_test, pred_clf_MNB,  colnames=["Classe réelle"], rownames=["Classe prédite"]))\nprint(classification_report(y_test,pred_clf_MNB))\nprint(" -- GradientBoosting -- ")\npred_clf_GB = clf_GB_unigram.predict(X_test)\ndisplay(pd.crosstab(y_test, pred_clf_GB,  colnames=["Classe réelle"], rownames=["Classe prédite"]))\nprint(classification_report(y_test,pred_clf_GB))')


# In[68]:


# #### Classification à l'aide des algorithmes : 
# - Régression logistique
# - SVM
# - Naïve Bayes
# - GradientBoosting
# Vectorisation prenant en compte des unigrams et bigrams


# In[69]:


get_ipython().run_cell_magic('time', '', '\ndf_train, df_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, random_state = 0)\n\nvec_ngram = TfidfVectorizer(analyzer=\'word\', ngram_range=(1,2)) #stopwords déjà supprimés dans le prétraitement\nX_train_text = vec_ngram.fit_transform(df_train.preprocessed)\nX_test_text = vec_ngram.transform(df_test.preprocessed)\n#on ajoute les metadonnées standardisées à notre vecteur d\'occurence\n\nX_train = hstack([X_train_text,df_train.drop("preprocessed", axis=1).values])\nX_test = hstack([X_test_text,df_test.drop("preprocessed", axis=1).values])\n\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)\n\n# instanciation des classifieurs \nclf_reglog = LogisticRegression(C=1, max_iter= 5000)\nclf_svc = SVC()\nclf_MNB= MultinomialNB() \nclf_GB = GradientBoostingClassifier()\n\n# Fit des classifieurs aux données d\'entraînement\nstart = time.time()\nclf_reglog.fit(X_train, y_train)\nend = time.time()\nprint("The time of clf_reglog_unigram:", end-start)\nstart = time.time()\nclf_svc.fit(X_train, y_train)\nend = time.time()\nprint("The time of clf_svc_unigram:", end-start)\nstart = time.time()\nclf_MNB.fit(X_train, y_train)\nend = time.time()\nprint("The time of clf_MNB_unigram:", end-start)\nstart = time.time()\nclf_GB.fit(X_train, y_train)\nend = time.time()\nprint("The time of clf_GB_unigram:", end-start)\n\n# Affichage des scores des différents classifieurs\n\nprint(" -- Régression logistique --")\nprint("Score sur le trainset :",clf_reglog.score(X_train,y_train),"; Score sur le testset : ",clf_reglog.score(X_test,y_test))\nprint(" -- SVC --")\nprint("Score sur le trainset :",clf_svc.score(X_train,y_train),"; Score sur le testset : ",clf_svc.score(X_test,y_test))\nprint(" -- Multinomial Naïve Bayes --")\nprint("Score sur le trainset :",clf_MNB.score(X_train,y_train),"; Score sur le testset : ",clf_MNB.score(X_test,y_test))\nprint(" -- GradientBoosting -- ")\nprint("Score sur le trainset :",clf_GB.score(X_train,y_train),"; Score sur le testset : ",clf_GB.score(X_test,y_test))')


# In[71]:


get_ipython().run_cell_magic('time', '', '# Affichage des prédictions par les différents classifieurs\nprint(" -- Régression logistique --")\npred_clf_reglog = clf_reglog.predict(X_test)\ndisplay(pd.crosstab(y_test, pred_clf_reglog,  colnames=["Classe réelle"], rownames=["Classe prédite"]))\nprint(classification_report(y_test,pred_clf_reglog))\nprint(" -- SVC --")\npred_clf_svc = clf_svc.predict(X_test)\ndisplay(pd.crosstab(y_test, pred_clf_svc,  colnames=["Classe réelle"], rownames=["Classe prédite"]))\nprint(classification_report(y_test,pred_clf_svc))\nprint(" -- Multinomial Naïve Bayes --")\npred_clf_MNB = clf_MNB.predict(X_test)\ndisplay(pd.crosstab(y_test, pred_clf_MNB,  colnames=["Classe réelle"], rownames=["Classe prédite"]))\nprint(classification_report(y_test,pred_clf_MNB))\nprint(" -- GradientBoosting -- ")\npred_clf_GB = clf_GB.predict(X_test)\ndisplay(pd.crosstab(y_test, pred_clf_GB,  colnames=["Classe réelle"], rownames=["Classe prédite"]))\nprint(classification_report(y_test,pred_clf_GB))')

# Mise au point du modèles Multinomial Naive Bayes - vectorisation TF-IDF Unigram
# In[419]:


get_ipython().run_cell_magic('time', '', 'from scipy.sparse import hstack\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.preprocessing import MaxAbsScaler\nscaler = MaxAbsScaler()\n\n\nvec_unigram = TfidfVectorizer(analyzer=\'word\', ngram_range=(1,1)) #stopwords déjà supprimés dans le prétraitement\nX_train_text = vec_unigram.fit_transform(df_train.preprocessed)\nX_test_text = vec_unigram.transform(df_test.preprocessed)\n#on ajoute les metadonnées standardisées à notre vecteur d\'occurence\n\nX_train = hstack([X_train_text,df_train.drop("preprocessed", axis=1).values])\nX_test = hstack([X_test_text,df_test.drop("preprocessed", axis=1).values])\n\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)')


# In[16]:


#GridSearchCV va nous permettre de tester une série de paramètres et de comparer 
#les performances pour en déduire le meilleur paramétrage.
from sklearn.model_selection import GridSearchCV
params = {"alpha": np.linspace(1e-10,1,30)}
grid_MN = GridSearchCV(MultinomialNB(), cv= 5, param_grid = params)
grid_MN.fit(X_train, y_train)
print("Meilleur paramètre : ", grid_MN.best_params_, "permettant d'obtenir un score de " , grid_MN.best_score_)
print("Score sur le trainset :",grid_MN.score(X_train,y_train),"; Score sur le testset : ",grid_MN.score(X_test,y_test))
print("On ne fait à priori pas face à un problème d'overfiting")


# In[17]:


# Best score training
import time

start = time.time()
clf_MNB_tfidf = MultinomialNB(alpha=0.5862068965931034)
clf_MNB_tfidf.fit(X_train,y_train)
end = time.time()

print("Durée d'entraînement :", end - start, "secondes")


# In[19]:


# Prédiction
from sklearn.metrics import classification_report
y_pred_MNB_tfidf_proba = clf_MNB_tfidf.predict_proba(X_test)
from sklearn.metrics import auc, roc_curve

fpr , tpr , seuil = roc_curve(y_test, y_pred_MNB_tfidf_proba[:,0], pos_label= 0)
aucf= auc(fpr, tpr)

plt.plot(fpr, tpr, color='coral', lw=2, label ='auc=%1.5f' % aucf)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("-- Multinomial Naïve Bayes ROC CURVE --")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - specificite (Taux de FN)', fontsize=14)
plt.ylabel('Sensibilite (Taux de VP)', fontsize=14)
plt.legend();
plt.show();

y_pred_MNB_tfidf = clf_MNB_tfidf.predict(X_test)
print(classification_report(y_test, y_pred_MNB_bow))

# Mise au point du modèle Regression Logistique - vectorisation TF-IDF Unigram
# In[24]:


get_ipython().run_cell_magic('time', '', 'from scipy.sparse import hstack\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.preprocessing import MaxAbsScaler\nscaler = MaxAbsScaler()\n\nvec_unigram = TfidfVectorizer(analyzer=\'word\', ngram_range=(1,1)) #stopwords déjà supprimés dans le prétraitement\nX_train_text = vec_unigram.fit_transform(df_train.preprocessed)\nX_test_text = vec_unigram.transform(df_test.preprocessed)\n#on ajoute les metadonnées standardisées à notre vecteur d\'occurence\n\nX_train = hstack([X_train_text,df_train.drop("preprocessed", axis=1).values])\nX_test = hstack([X_test_text,df_test.drop("preprocessed", axis=1).values])\n\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)')


# In[277]:





# In[52]:


#GridSearchCV va nous permettre de tester une série de paramètres et de comparer 
#les performances pour en déduire le meilleur paramétrage.
# define models and parameters
model = LogisticRegression(max_iter= 500)
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
C = [100, 10, 1.0, 0.1, 0.01]
grid = dict(solver=solvers,penalty=penalty,C=C)
# define grid search
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=5, scoring='accuracy')
grid_lr = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_lr.best_score_, grid_lr.best_params_))
means = grid_lr.cv_results_['mean_test_score']
stds = grid_lr.cv_results_['std_test_score']
params = grid_lr.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[25]:


# Best score training
from joblib import dump, load
import time

start = time.time()
clf_lr_tfidf = LogisticRegression(max_iter= 500, C = 1.0, penalty = 'l2', solver = 'liblinear')
clf_lr_tfidf.fit(X_train,y_train)
end = time.time()

print("Durée d'entraînement :", end - start, "secondes")


# In[26]:


# Prédiction
from sklearn.metrics import classification_report
y_pred_lr_tfidf_proba = clf_lr_tfidf.predict_proba(X_test)
from sklearn.metrics import auc, roc_curve

fpr , tpr , seuil = roc_curve(y_test, y_pred_lr_tfidf_proba[:,0], pos_label= 0)
aucf= auc(fpr, tpr)

plt.plot(fpr, tpr, color='coral', lw=2, label ='auc=%1.5f' % aucf)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("-- Multinomial Naïve Bayes ROC CURVE --")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - specificite (Taux de FN)', fontsize=14)
plt.ylabel('Sensibilite (Taux de VP)', fontsize=14)
plt.legend();
plt.show();

y_pred_lr_tfidf = clf_lr_tfidf.predict(X_test)
print(classification_report(y_test, y_pred_lr_tfidf))


# # inclure ici la joblib 

# In[27]:


##### construction pipeline


# In[28]:


### fonction de nétoyage du commentaire
from scipy.sparse import hstack
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_comment(x):
    x = " ".join(x for x in str(x).strip().split())
    x = remplacement_carac_e(x)
    x = neg_identify(x)
    x = stopwords_filtering(x)
    x = stemming(x).strip()
    return (x)

from sklearn.feature_extraction.text import TfidfVectorizer
vec_unigram = TfidfVectorizer(analyzer='word', ngram_range=(1,1))

def comment_preprocessed (x,y):
    concat_vector = hstack([vector, y])
    return concat_vector


# In[29]:


clean_comment("je suis content de cet achat !")


# In[36]:


from sklearn.base import TransformerMixin, BaseEstimator
class comment_cleaner(BaseEstimator, TransformerMixin ):
    def transform(self, X, y=None , **fit_params):
        return [clean_comment(comment) for comment in X]
    # just return self
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params (self, deep = True):
        return {}


# In[48]:


from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

pipe_tfidf_lr = Pipeline(steps=[
    ('clean_comment',comment_cleaner()),
    ('vectorizer', vec_unigram),
    ('classifieur',clf_lr_tfidf)], 
    verbose= True)
# convert df to array 

df_train = np.array(df_train)

# pipeline training

pipe_tfidf_lr.fit(df_train.reshape(-1,1),y_train)

df_test = np.array(df_test)

pipe_tfidf_lr.predict(df_test)

# Save to file in the current working directory
joblib_file = "pipe_tfidf_lr.pkl"
joblib.dump(pipe_tfidf_lr, joblib_file)


# In[46]:



pipe_tfidf_lr.score(df_test, y_test)


# In[ ]:





# In[47]:


pipe_tfidf_lr = joblib.load("pipe_tfidf_lr_.joblib")
pipe_tfidf_lr


# In[ ]:





# In[ ]:





# In[ ]:




