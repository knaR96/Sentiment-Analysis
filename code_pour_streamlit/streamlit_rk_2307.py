# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 12:48:33 2022

@author: kevin
"""
#########################################################


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


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

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, random_state = 0)


# In[18]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB #Particulièrement adapté aux problèmes de classification avec des features discretes (text classification)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report



# In[52]:

# In[25]:


# Best score training
from joblib import dump, load
import time

start = time.time()
clf_lr_tfidf = LogisticRegression(max_iter= 500, C = 1.0, penalty = 'l2', solver = 'liblinear')

end = time.time()

print("Durée d'entraînement :", end - start, "secondes")




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
import joblib

pipe_tfidf_lr = Pipeline(steps=[
    ('clean_comment',comment_cleaner()),
    ('vectorizer', vec_unigram),
    ('classifieur',clf_lr_tfidf)], 
    verbose= True)
# convert df to array 

X_train = np.array(X_train)

# pipeline training

pipe_tfidf_lr.fit(X_train.reshape(-1,1),y_train)

X_test = np.array(X_test)

pipe_tfidf_lr.predict(df_test)

# Save to file in the current working directory
joblib_file = "C:/Users/kevin/0_projet_satisfaction_client/10_Streamlit/code_pour_streamlit/joblib_file/pipe_tfidf_lr.pkl"
joblib.dump(pipe_tfidf_lr, joblib_file)


#########################################################

##### import visualisation
from locale import normalize
import streamlit as st
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize.regexp import RegexpTokenizer
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
######################
### import partie modélisation


import streamlit as st
import time
import numpy as np
import pandas as pd
import pyarrow.lib as _lib
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
import pyarrow.lib as _lib
import plotly.express as px
from matplotlib import cm


df = pd.read_csv("/Users/kevin/0_projet_satisfaction_client/10_Streamlit/reviews_trust.csv")
best_models = pd.read_excel(r'C:\Users\kevin\0_projet_satisfaction_client\10_Streamlit\best_models.xlsx')
best_models = best_models.set_index(["MODELES"])  
#Import tableau résultats
table_score= pd.read_excel(r'C:\Users\kevin\0_projet_satisfaction_client\10_Streamlit\table_score.xlsx')
table_score = table_score.astype("str")
# Fonction de prediction tfidf_lr


pipe_tfidf_lr = joblib.load(r"C:/Users/kevin/0_projet_satisfaction_client/10_Streamlit/code_pour_streamlit/joblib_file/pipe_tfidf_lr.pkl")
    

def prediction_tfidf_lr(comment):
    results = pipe_tfidf_lr.predict([comment])
    return results[0]

def prediction_tfidf_lr_proba(comment):
    results = pipe_tfidf_lr.predict_proba([comment])
    return results



# config position text streamlit
st.set_page_config(layout="wide")

def main():
 

    st.title("CustomerSatisf_survey")
    menus = ["Accueil","Introduction", "🔍 Exploration des données", "📈 Modélisation", "🪄 Démonstration"]
    choice= st.sidebar.radio(" Navigation 🧭 ",options=menus)
    st.markdown("---")
    
     ### Data import 
       
    
    if choice == "Accueil":
        st.header("Application d'analyse de sentiments de commentaires")
        
        
    if choice == "Introduction":
        st.subheader("Introduction")
        
       
    elif choice == "🔍 Exploration des données":
        #this is the header
        t1, t2 = st.columns((0.07,1))
        t2.title("Exploration et visualisation des données ")
        st.markdown("-----------------------------")
        
        ## Data#
        with st.spinner('Updating Report...'):
            #Metrics setting and rendering
            df = pd.read_csv("/Users/kevin/0_projet_satisfaction_client/10_Streamlit/reviews_trust.csv")
            m1, m2, m3, m4 = st.columns((1,1,1,1))  
            m1.write('')
            m2.metric(label ='Nombre de commentaires',value = df.shape[0], delta_color = 'inverse')
            m3.metric(label ="Nombre d'étoile",value = df.shape[1], delta_color = 'inverse')
            m4.write('')
            cw1, cw2 = st.columns((6.5, 1.7))  
        row3_spacer1, row3_1, row3_spacer2 = st.columns((.2, 7.1, .2))
        with row3_1:
            st.markdown("")
            affichage_df = st.expander('Cliquez ici pour faire apparaître le jeu de données 👉🏼 ')
            with affichage_df:
                st.dataframe(data=df.reset_index(drop=True))
                st.text('')
               
        st.markdown("-----------------------------")              
        g1, g2 = st.columns((1,1))
        mean_counts_by_star = df.groupby('company').mean()[['star']].reset_index()
        fig = px.bar(mean_counts_by_star, x='star', y='company',color='star', template = 'seaborn', color_discrete_sequence=px.colors.sequential.OrRd)
        # fig.update_traces(marker_color='#264653')
        fig.update_layout(title_text="Distribution des notes moyennes par compagnies",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None, xaxis = (dict(showgrid = False)),yaxis = (dict(showgrid = False)), legend=dict(orientation="h",yanchor="bottom",y=0.7,xanchor="right",x=0.1))
        g1.plotly_chart(fig, use_container_width=True)
             
        mean_counts_by_star = df.groupby('source').mean()[['star']].reset_index()
        fig = px.bar(mean_counts_by_star, x='star', y='source',color='star', template = 'seaborn', color_discrete_sequence=px.colors.sequential.OrRd) #, color_discrete_sequence = ["blu","yellow"])
        fig.update_layout(title_text="La moyenne entre les étoiles et la source( les sites web)",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None, xaxis = (dict(showgrid = False)),yaxis = (dict(showgrid = False)), legend=dict(orientation="h",yanchor="bottom",y=0.1,xanchor="right",x=0.1))
        g2.plotly_chart(fig, use_container_width=True)
        
        
        #correlation map#
        g3, g4 = st.columns((2,2))
        
        st.markdown("-----------------------------")
        
        
        # la distribution des étoiles
        g4, g5 = st.columns((2,2))
        x= df.star.value_counts()
        fig = px.bar(x, template = 'seaborn', color = x, color_discrete_sequence = ["orange","brown", "green", "darkslateblue",'grey'], color_continuous_scale=px.colors.sequential.OrRd)
        fig.update_layout(title_text="Distribution des étoiles (en nombre absolus)",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None, xaxis = (dict(showgrid = False)),yaxis = (dict(showgrid = False)), legend=dict(orientation="h",yanchor="bottom",y=0.1,xanchor="right",x=0.1))
        g4.plotly_chart(fig, use_container_width=True)
        
        
        
        
        pie_col, donut=st.columns(2)
        fig = px.pie(df, values='star', names='star', title='Distribution des étoiles', color_discrete_sequence=px.colors.sequential.OrRd)
        pie_col.plotly_chart(fig, use_container_width=True)
        
        labels = ["Star: " + str(i) for i in df.star.value_counts().index.tolist()]
        values = df.star.value_counts().values.tolist()
        
        st.markdown("-----------------------------")
        
        
        st.markdown("<h3 style='text-align: left; color:white;'>Les distribution a travers le temps par mois</h3>", unsafe_allow_html=True)
        g9, g10 = st.columns((4.5,0.5))
        x = df[['date', 'star']]
        x['date'] = x.date.astype(str).apply(lambda x: x[:10])
        x['date'] = pd.to_datetime(x.date)
        x['month'] = x['date'].dt.strftime('%B')
        x['digitmonth'] = x['date'].dt.strftime('%m')
        x = x.groupby(['digitmonth', 'month']).count().reset_index().rename(columns = {'star' : 'Nombre de commentaires'})
        x.drop(['digitmonth'], axis=1, inplace=True)
        fig = px.bar(x, x=x.month, y='Nombre de commentaires', color = 'Nombre de commentaires', template = 'seaborn', color_discrete_sequence=px.colors.sequential.OrRd)
        fig.update_layout(title_text="Nombre de commentaires par Mois",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None, xaxis = (dict(showgrid = False)),yaxis = (dict(showgrid = False)), legend=dict(orientation="h",yanchor="bottom",y=0.1,xanchor="right",x=0.1))
        g9.plotly_chart(fig, use_container_width=True)
        
        
        g11, g12 = st.columns((4.5,0.5))
        x = df[['date', 'star']]
        x['date'] = x.date.astype(str).apply(lambda x: x[:10])
        x['date'] = pd.to_datetime(x.date)
        x['month'] = x['date'].dt.strftime('%B')
        x['digitmonth'] = x['date'].dt.strftime('%m')
        x = x.groupby(['digitmonth', 'month']).mean().reset_index().rename(columns = {'star' : 'Moyenne des star'})
        x.drop(['digitmonth'], axis=1, inplace=True)
        fig = px.bar(x, x=x.month, y='Moyenne des star', color = 'Moyenne des star', template = 'seaborn', color_discrete_sequence=px.colors.sequential.OrRd)
        fig.update_layout(title_text="Moyenne des star par Mois",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None, xaxis = (dict(showgrid = False)),yaxis = (dict(showgrid = False)), legend=dict(orientation="h",yanchor="bottom",y=0.1,xanchor="right",x=0.1))
        g11.plotly_chart(fig, use_container_width=True)
        
        
        st.markdown("-----------------------------")
        
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Create some sample text
        
        #df = pd.read_csv(r"C:\Users\selma\Desktop\Projet satisfaction Client\reviews_trust.csv")  
        
        # Definition d'une fonction de filtrage de stopwords
        def stopwords_filtering(chaine):
            tokenizer = RegexpTokenizer("[a-zA-Zéèê]{3,}")  
            # df["preprocessed"] = df["Comme?ntaire"].apply(lambda x : " ".join(x.lower() for x in str(x).strip().split()))
        
            chaine = str(chaine).lower()
            stop_words = set(stopwords.words('french'))
            stop_words.update (["!","?",".","_",":",",",";","-","--","...","'","...","'",',',',','…la','la','le','les','..','…','(',')','a+','+','etc…','qq','``',"j'","j '"])
            tokens =[]
            chaine = tokenizer.tokenize(chaine)
        
            for mot in chaine:
                if mot not in stop_words :#conservation des mots non stopwords
                    tokens.append(mot)
            tokens = ",".join(tokens)
            return tokens
        
        df["preprocessed"] = df["Commentaire"].apply(lambda x : stopwords_filtering(x))
        commentaires_negatifs = ",".join(i for i in df.preprocessed[df["star"] < 3])
        commentaires_positifs = ",".join(i for i in df.preprocessed[df["star"] >= 3])
        text=commentaires_negatifs
        # Create and generate a word cloud image:
        wordcloud = WordCloud().generate(text)
        # Display the generated image:
        fig=plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.pyplot()
        
        text=commentaires_positifs
        # Create and generate a word cloud image:
        wordcloud = WordCloud().generate(text)
        # Display the generated image:
        fig=plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot()
       
    elif choice == "📈 Modélisation":
        
        row_space1, row, row_space2 = st.columns((6, 6, 6))
        with row:
            st.subheader("Méthodologie")
            st.text('')
        
        st.subheader('Prétraitement des données')
        image = Image.open("/Users/kevin/0_projet_satisfaction_client/10_Streamlit/pipeline_preprocessing.png")
        st.image(image, caption='Pipeline de prétraitement des données',width =1200)
        st.text('')
        st.text('')
        st.markdown("---")
    #####################################################################
        row4_spacer1, row4_1, row4_spacer2 = st.columns((.7, 7, .7))
        with row4_1:        
            st.subheader("APPROCHES PROPOSEES")
            st.markdown(" ")
            image = Image.open("/Users/kevin/0_projet_satisfaction_client/10_Streamlit/methode.png")
            st.image(image)
     ####################################################################
           
        row4_spacer1, row, row4_spacer2 = st.columns((.7, 7, .7))   
        with row:        
            st.markdown(" Méthode de traitement des commentaires ")
            st.text("1) Commentaires considérés dans leur entièreté \n2) Commentaires découpés en phrases" )
            
        row4_spacer1, row, row4_spacer2 = st.columns((.7, 7, .7))   
        with row:        
            st.markdown(" Déclinaison des méthodes par comptage directe : BOW et TF-IDF ")
            st.text("1) Vectorisation générant uniquement des unigrammes \n2) Vectorisation générant des unigrammes et bigrammes")
    
    
    ### import tableau table_score
            st.markdown("")
            affichage_tableau= st.expander('Cliquez ici pour visualiser le tableau des résultats 👉🏼 ')
            with affichage_tableau:
                st.dataframe(table_score)
                st.text('')
    
    ####################################################################
        st.markdown("---")
        ### Affichage des scores des 4 meilleurs modèles###
        
        row4_spacer1, row4_1, row4_spacer2 = st.columns((.2, 7.1, .2))
        with row4_1:
            st.subheader("MEILLEURS MODELES")
            st.text('')  
        row5_spacer1, row5_1, row5_spacer2  = st.columns((.2, 7.1, .2))
        with row5_1:
            st.markdown("Affichage des scores et durée d'entrainement des meilleurs modèles")
            
            ### création du dataframe des scores ###
            
            st.markdown("")
            affichage_best_models= st.expander('Tableau comparatif des cinq meilleurs modèles 👉🏼 ')
            with affichage_best_models:
                st.dataframe(best_models)
                st.text('')
            
            #Accuracy plot 
            accuracy_score_df = best_models["Accuracy"]
            
            accuracy_chart = px.bar(
                                    accuracy_score_df,
                                    x = accuracy_score_df.index,
                                    y = "Accuracy",
                                    text_auto='.2',
                                    color = best_models.index,
                                    color_discrete_sequence = ['#000c88',"#006d00", "#a60000", "#ba5b00",'#e39e0e'],
                                    title= "<b> Accuracy score <b>",
                                    )
            accuracy_chart.update_layout(
                xaxis = dict(tickmode = 'linear'),
                plot_bgcolor = "rgba(0,0,0,0)",
                yaxis = (dict(showgrid = False)),
                showlegend = False
                )
            
            accuracy_chart.update_traces(textfont_size=15,textfont_color='white',marker=dict(line=dict(color='black', width=1)))
            
            
            #Durée d'entraînement plot 
            
            duration_df = best_models["Duree d'entrainement (s)"]
            
            Duration_chart = px.bar(
                                    duration_df,
                                    x = accuracy_score_df.index,
                                    y = "Duree d'entrainement (s)",
                                    range_y = [0,5000],
                                    text_auto='.f',
                                    color = best_models.index,
                                    color_discrete_sequence = ['#000c88',"#006d00", "#a60000", "#ba5b00",'#e39e0e'],
                                    title= "<b> Durée d'entraînemet <b>",
                                    
                                    )
            Duration_chart.update_layout(
                xaxis = dict(tickmode = 'linear'),
                plot_bgcolor = "rgba(0,0,0,0)",
                yaxis = (dict(showgrid = False)),
                showlegend = False
                )
            Duration_chart.update_traces(textfont_size=16,textfont_color='white',textposition='outside',marker=dict(line=dict(color='black', width=1)))
            
            
            
            #F1_score plot 
            
            F1_score_df = best_models[["F1-Score_classe 0","F1-Score_classe 1"]]
            
            F1_score_chart = px.bar(F1_score_df, x=F1_score_df.index, y=["F1-Score_classe 0","F1-Score_classe 1"], title="<b> F1-Score <b>",
                                    color_discrete_sequence = ["#000c88","#a60000"],
                                    text_auto='.2', 
                                    range_y = [0,1.5],
                                    )
            # build from two different figures
            F1_score_chart.update_layout(barmode = 'group',  yaxis = (dict(showgrid = False)),xaxis = dict(tickmode = 'linear'),
                plot_bgcolor = "rgba(0,0,0,0)",)
            F1_score_chart.update_traces(textfont_size=16,textfont_color='white',textposition='outside',marker=dict(line=dict(color='black', width=1)))
        
        ### plot des charts 
        
        col1, col2 = st.columns(2)
        with col1 :
            st.write(accuracy_chart)
        with col2 :
            st.write(Duration_chart)
        col3, col4, col5 = st.columns([4, 6, 6])
        with col3:
            st.write(' ')
        with col4:
            st.write(F1_score_chart)
        with col5:
            st.write(' ')
            
    ######  
        
        st.markdown(" ")
        st.markdown("---")
        st.subheader("MODELE RETENU POUR LA MISE EN PRODUCTION")
    
    
    ################################################ Demonstration
    elif choice ==  "🪄 Démonstration" :
        st.subheader("Analyse de sentiments")  
        with st.form(key='detection_de_sentiment'):
            commentaire = st.text_area("Commentaire client")
            bouton_classification = st.form_submit_button(label="Classifier")
            
                                
    
    
        
        if bouton_classification : 
            col1, col2 = st.columns(2)
            
            
            # application des fonction 
            prediction = prediction_tfidf_lr(["raw_test"])
            probability = prediction_tfidf_lr_proba("raw_test")       
            
            with col1:
                    st.success("Commentaire client")
                    st.write(commentaire)
                    st.success("Prediction")
                    st.write(prediction)
                    
            with col2:
                    st.success("Niveau de confiance")
                    st.write(probability)
                    
if __name__ == '__main__':
  main()                           
            