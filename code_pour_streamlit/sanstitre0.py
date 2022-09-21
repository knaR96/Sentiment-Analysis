# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 15:28:08 2022

@author: kevin
"""


### définition de la classe comment_preprocessed

# On remplace les éèê par e
def remplacement_carac_e(com):
    text=re.sub(r"[éèêë]","e",com)
    return text

def neg_identify(text): 
    text = tokenizer.tokenize(text)
    for pos, word in enumerate(text) :
        if pos != len(text)-1 :
            if (word == 'ne' or word == "n" or word =="n'"):
                text[pos+1] = "NON_"+text[pos+1]
    text = " ".join(text)
    return (text)


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

def clean_comment(x):
    x = " ".join(x for x in str(x).strip().split())
    x = remplacement_carac_e(x)
    x = neg_identify(x)
    x = stopwords_filtering(x)
    x = stemming(x).strip()
    return (x)


from sklearn.base import TransformerMixin, BaseEstimator
class comment_cleaner(BaseEstimator, TransformerMixin ):
    def transform(self, X, y=None , **fit_params):
        return [clean_comment(comment) for comment in X]
    # just return self
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params (self, deep = True):
        return {}