#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec


class GensimWord2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Word vectors are averaged across to create the document-level vectors/features.
    gensim's own gensim.sklearn_api.W2VTransformer doesn't support out of vocabulary words,
    hence we roll out our own.
    All the parameters are gensim.models.Word2Vec's parameters.
    https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
    """

    def __init__(self, vector_size=200, # nombre de variables indépendantes souhaité
            window=5, # nombre de fenêtres # initialement 5
            min_count=2, # Ignorer les mots dont la fréquence totale est inférieure à 2                                 
            sg = 1, # 1 pour skip-Gram
            hs = 0,
            negative = 10, # for negative sampling
            workers= -1, # no.of cores
            seed = 34):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.seed = seed
        self.workers = workers
        self.sg = sg
        self.hs = hs
        self.negative = negative
      
    def fit(self, X, y=None):
        self.model_ = Word2Vec(
            sentences=X, corpus_file=None,
            vector_size=self.vector_size, window=self.window, min_count=self.min_count,
            seed=self.seed,
            workers=self.workers,sg=self.sg, hs=self.hs,
            negative=self.negative)
        return self

    def transform(self, X):
        X_embeddings = np.array([self._get_embedding(words) for words in X])
        return X_embeddings

    def _get_embedding(self, words):
        valid_words = [word for word in words if word in self.model_.wv.index_to_key ]
        if valid_words:
            embedding = np.zeros((len(valid_words), self.vector_size), dtype=np.float32)
            for idx, word in enumerate(valid_words):
                embedding[idx] = self.model_.wv[word]

            return np.mean(embedding, axis=0)
        else:
            return np.zeros(self.vector_size)


# In[ ]:





# In[ ]:




