# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 12:58:26 2022

@author: kevin
"""



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
from preprocessed_class_tfidf import comment_cleaner

# config position text streamlit

st.set_page_config(layout="wide")

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

#####################################################
#importation des modèles 

import joblib
pipe_tfidf_lr = joblib.load(r"C:/Users/kevin/0_projet_satisfaction_client/10_Streamlit/code_pour_streamlit/joblib_file/pipe_tfidf_lr_.pkl")
pipe_tfidf_MNB = joblib.load(r"C:/Users/kevin/0_projet_satisfaction_client/10_Streamlit/code_pour_streamlit/joblib_file/pipe_tfidf_MNB_.pkl")
pipe_BOW_MNB = joblib.load(r"C:/Users/kevin/0_projet_satisfaction_client/10_Streamlit/code_pour_streamlit/joblib_file/pipe_BOW_MNB.pkl")

def prediction_tfidf_lr(comment):
    results = pipe_tfidf_lr.predict([comment])
    return results[0]

def prediction_tfidf_lr_proba(comment):
    results = pipe_tfidf_lr.predict_proba([comment])
    return results

def prediction_tfidf_MNB(comment):
    results = pipe_tfidf_MNB.predict([comment])
    return results[0]

def prediction_tfidf_MNB_proba(comment):
    results = pipe_tfidf_MNB.predict_proba([comment])
    return results

def prediction_BOW_MNB(comment):
    results = pipe_BOW_MNB.predict([comment])
    return results[0]

def prediction_BOW_MNB_proba(comment):
    results = pipe_BOW_MNB.predict_proba([comment])
    return results

#### Affichage des graphs de distribution des métadonnées

df_viz = df[["Commentaire", "star"]]
df_viz["Sentiment"] = df["star"].apply(lambda x : np.where(x >=4 , 1 , 0))  # ajout de la colonne Sentiment à df

#Suppression des valeurs manquantes de la colonne Commentaire
df_viz = df_viz.dropna(axis = 0, how = 'any', subset =["Commentaire"])
df_viz.reset_index(inplace = True)
df_viz = df_viz.drop(['index','star'],axis = 1)

#Nettoyages des commentaires et étapes de normalisation 

def find_exclamation(com):   #compte le nombre de points d'exclamation d'un commentaire
    r = re.compile(r"\!")
    exclamation = r.findall(com)
    return len(exclamation)

def find_interogation(com): #compte le nombre de points d'interogation d'un commentaire
    r = re.compile(r"\?")
    interogation = r.findall(com)
    return len(interogation)

def findCAPSLOCK(com):  #compte le nombre de caractères en majuscule d'un commentaire
    r = re.compile(r"[A-Z]")
    capslock = r.findall(com)
    return len(capslock)

def find_etc(com): #compte le nombre de chaine de ".." d'un commentaire
    r = re.compile(r"\.{2,}")
    etc = r.findall(com)
    return len(etc)

df_viz["CAPSLOCK"]= df_viz["Commentaire"].apply(lambda x : findCAPSLOCK(x))
df_viz["exclamation"]= df_viz["Commentaire"].apply(lambda x : find_exclamation(x))
df_viz["interrogation"]= df_viz["Commentaire"].apply(lambda x : find_interogation(x))
df_viz["chainpoints"]= df_viz["Commentaire"].apply(lambda x : find_etc(x))
df_viz['nb_caracter'] = df_viz["Commentaire"].apply(len)

df_viz_bad = df_viz[df_viz["Sentiment"]<= 3]
df_viz_good = df_viz[df_viz["Sentiment"]> 3]
######
#wordcloud avec masque

from PIL import Image
import numpy as np


#########



def main():
    
    #style de hide menu
    hidemenu= """<style>
    footer{visibility:visible;}
    footer:before{content:'Nacira IKHENECHE \A Kevin ROGER \A Selma BENABID';white-space:pre;display:block;position:relative;}
    </style>"""
    
    footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: black;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p> Created by <a style='display: block; text-align: center;' target="_blank">Nacira IKHENECHE  -  Kevin ROGER - Selma BENABID </a></p>
</div>
"""
    
    st.sidebar.header(" ")

    with st.sidebar.container():
        image = Image.open(r"C:\Users\kevin\0_projet_satisfaction_client\10_Streamlit\satisfaction_client_image.png")
     
        st.image(image, use_column_width=True)
        
        st.markdown(" ")
        st.markdown(" ")
        
    menus = ["Accueil", "Exploration des données", "Modélisation", "Démonstration"]
    choice= st.sidebar.radio(" Navigation ",options=menus)
    
     ### Data import 
    
    if choice == "Accueil":

        st.title("Application pour l'analyse de sentiments de commentaires clients")
        
        
        cloud=Image.open(r'C:/Users/kevin/0_projet_satisfaction_client/10_Streamlit/nuage.PNG')
        
        row4_spacer1, row, row4_spacer2 = st.columns((.7, 7, .7))

        with row:
            st.markdown(" ")
            st.markdown(" ")
            st.markdown(" ")
           
            placeholder = st.empty()
            placeholder.image(cloud,width=1050)      
        
                        
    elif choice == "Exploration des données":
        #this is the header
        t1, t2 = st.columns((0.07,1))
        t2.title("Exploration et visualisation des données ")
        st.markdown("-----------------------------")
        
        ## Data#
        with st.spinner('Updating Report...'):
            #Metrics setting and rendering
            df = pd.read_csv(r"C:\Users\kevin\0_projet_satisfaction_client\codes_finaux\reviews_trust.csv")
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
                

        
        #g1, g2 = st.columns((4.5,0.5))
        x = df[['date', 'star']]
        x['date'] = x.date.astype(str).apply(lambda x: x[:10])
        x['date'] = pd.to_datetime(x.date)
        x['month'] = x['date'].dt.strftime('%B')
        x['digitmonth'] = x['date'].dt.strftime('%m')
        x = x.groupby(['digitmonth', 'month']).count().reset_index().rename(columns = {'star' : 'Nombre de commentaires'})
        x.drop(['digitmonth'], axis=1, inplace=True)
        fig = px.bar(x, x=x.month, y='Nombre de commentaires', color = 'Nombre de commentaires', template = 'seaborn', color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_layout(title_text="Nombre de commentaires par mois",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None, xaxis = (dict(showgrid = False)),yaxis = (dict(showgrid = False)), legend=dict(orientation="h",yanchor="bottom",y=0.1,xanchor="right",x=0.1))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("-----------------------------")

        g1, g2 = st.columns((1,1))
        mean_counts_by_star = df.groupby('company').mean()[['star']].reset_index()
        fig = px.bar(mean_counts_by_star, x='star', y='company',color='star', template = 'seaborn', color_discrete_sequence=px.colors.sequential.RdBu)
        # fig.update_traces(marker_color='#264653')
        
        fig.update_layout(title_text="Distribution des notes moyennes par compagnies",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None, xaxis = (dict(showgrid = False)),yaxis = (dict(showgrid = False)), legend=dict(orientation="h",yanchor="bottom",y=0.7,xanchor="right",x=0.1))
        g1.plotly_chart(fig, use_container_width=True)
             
        mean_counts_by_star = df.groupby('source').mean()[['star']].reset_index()
        fig = px.bar(mean_counts_by_star, x='star', y='source', color_discrete_sequence=px.colors.sequential.RdBu) #, color_discrete_sequence = ["blu","yellow"])
        fig.update_layout(title_text="Distribution des notes moyennes par rapport aux sources",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None, xaxis = (dict(showgrid = False)),yaxis = (dict(showgrid = False)), legend=dict(orientation="h",yanchor="bottom",y=0.1,xanchor="right",x=0.1))
        g2.plotly_chart(fig, use_container_width=True)

        st.markdown("-----------------------------")
        
        # la distribution des étoiles
        row5_spacer1,row5_spacer2  = st.columns(2)
        with row5_spacer1:
            x= df.star.value_counts()
            fig = px.bar(x, color = x, template = 'seaborn', color_discrete_sequence = ["orange","brown", "green", "darkslateblue",'grey'], color_continuous_scale=px.colors.sequential.RdBu)
            fig.update_layout(title_text="Distribution des notes selon les classes",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None, xaxis = (dict(showgrid = False)),yaxis = (dict(showgrid = False)), legend=dict(orientation="h",yanchor="bottom",y=0.1,xanchor="right",x=0.1))
            st.plotly_chart(fig, use_container_width=True)
        with row5_spacer2:
            fig = px.pie(df, values='star', names='star', title='', color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)
            labels = ["Star: " + str(i) for i in df.star.value_counts().index.tolist()]
            values = df.star.value_counts().values.tolist()
        

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
        commentaires_negatifs = ",".join(i for i in df.preprocessed[df["star"] <= 3])
        commentaires_positifs = ",".join(i for i in df.preprocessed[df["star"] > 3])
        # Create and generate a word cloud image:

        
        
 #######
#graph binaire           
        df = df[["Commentaire", "star"]]
        df["Sentiment"] = df["star"].apply(lambda x : np.where(x >=4 , 1 , 0))
        sns.countplot(df.Sentiment);
        sns.set(rc={'figure.figsize':(10,10)})
        x= df["Sentiment"].value_counts(normalize = True)
        fig = px.bar(x, template = 'seaborn', color = x , color_discrete_sequence=['indigo','firebrick'])
        fig.update_layout(title_text="Distribution des sentiments, représentation binaire",title_x=0,margin= dict(l=0,r=10,b=10,t=30), yaxis_title=None, xaxis_title=None, xaxis = (dict(showgrid = False)),yaxis = (dict(showgrid = False)), legend=dict(orientation="h",yanchor="bottom",y=0.1,xanchor="right",x=0.1))
        st.plotly_chart(fig, use_container_width=True)
      
######
#wordcloud avec masque
        def plot_word_cloud_pos(text, masque, background_color = "black") :
            # Définir un masque
            mask_coloring = np.array(Image.open(str(masque)))
        
            # Définir le calque du nuage des mots
            wc = WordCloud(background_color=background_color, max_words=150, mask = mask_coloring, max_font_size=25, random_state=42)
        
            # Générer et afficher le nuage de mots
            plt.figure(figsize= (7,5))
            wc.generate(text)
            plt.axis("off")
            plt.imshow(wc)
            
            plt.show()
            
        def plot_word_cloud_neg(text, masque, background_color = "black") :
            # Définir un masque
            mask_coloring = np.array(Image.open(str(masque)))
        
            # Définir le calque du nuage des mots
            wc = WordCloud(background_color=background_color, max_words=150, mask = mask_coloring, max_font_size=29, random_state=42)
        
            # Générer et afficher le nuage de mots
            plt.figure(figsize= (10,7))
            wc.generate(text)
            plt.axis("off")
            plt.imshow(wc)
            
            plt.show()
        st.write("---")
        
        g4, g5 = st.columns((1,1))
        with g4:
            
            st.subheader("Nuage de mots des commentaires positifs") 
            wc_good_comments = plot_word_cloud_pos(commentaires_positifs, "C:/Users/kevin/0_projet_satisfaction_client/10_Streamlit/pouce_bon.png")
            st.pyplot(wc_good_comments)  
            
        
        with g5:
            st.subheader("Nuage de mots des commentaires négatifs") 
            st.markdown(" ")
            
            wc_bad_comments = plot_word_cloud_neg(commentaires_negatifs, "C:/Users/kevin/0_projet_satisfaction_client/10_Streamlit/pouce_mauvais.png")
            st.pyplot(wc_bad_comments)
        
        st.write("---")
        
#### distribution des caractères spéciaux
        g4, g5 = st.columns((1,1))
        with g4:
            fig = px.histogram(df_viz, x="CAPSLOCK", color="Sentiment", range_x = (0,30), color_discrete_sequence=['indigo','firebrick'],title ="Distribution du nombre de caractères Majuscule selon les classes",
                                     hover_data=df_viz.columns)
            fig.update_layout(xaxis = (dict(showgrid = False)),yaxis = (dict(showgrid = False)))
            st.write(fig)
        with g5:
            fig = px.histogram(df_viz, x="exclamation", color="Sentiment", range_x = (0,10), color_discrete_sequence=['indigo','firebrick'],title ="Distribution du nombre de points d'exclamation selon les classes",
                                     hover_data=df_viz.columns)
            fig.update_layout(xaxis = (dict(showgrid = False)),yaxis = (dict(showgrid = False)))
            st.write(fig)
            
        g6, g7 = st.columns((1,1))
        with g6:
            fig = px.histogram(df_viz, x="interrogation", color="Sentiment", range_x = (0,10), color_discrete_sequence=['indigo','firebrick'],title ="Distribution du nombre de points d'interrogation selon les classes",
                                     hover_data=df_viz.columns)
            fig.update_layout(xaxis = (dict(showgrid = False)),yaxis = (dict(showgrid = False)))
            st.write(fig)
        with g7:
            fig = px.histogram(df_viz, x="chainpoints", color="Sentiment", range_x = (0,10), color_discrete_sequence=['indigo','firebrick'],title ="Distribution du nombre de points de suspension selon les classes",
                                     hover_data=df_viz.columns)
            fig.update_layout(xaxis = (dict(showgrid = False)),yaxis = (dict(showgrid = False)))
            st.write(fig)
        
        fig = px.histogram(df_viz, x="nb_caracter", color="Sentiment", range_x = (0,750), color_discrete_sequence=['indigo','firebrick'],title ="Distribution du nombre de caractères selon les classes",
                                     hover_data=df_viz.columns)
        fig.update_layout(xaxis = (dict(showgrid = False)),yaxis = (dict(showgrid = False)))
        st.write(fig)

#########
      
    elif choice == "Modélisation":
        option=st.selectbox(" ",('Sélectionnez le paragraphe souhaité','PRETRAITEMENT DES DONNEES', 'APPROCHES PROPOSEES',"MEILLEURS MODELES","MODELE RETENU POUR LA MISE EN PRODUCTION"))
        
        if option == 'Selectionnez le paragraphe souhaité':
            pass
        
        if option == 'PRETRAITEMENT DES DONNEES' :
 
            #####################################################################
            row4_spacer1, row4_1, row4_spacer2 = st.columns((.7, 7, .7))
            with row4_1: 
                image = Image.open("/Users/kevin/0_projet_satisfaction_client/10_Streamlit/pipeline_preprocessing.png")
                st.image(image, caption='Pipeline de prétraitement des données',width =1150)
                st.text('')
                st.text('')
                st.markdown("---")
    #####################################################################
        if option == 'APPROCHES PROPOSEES' :    
            row4_spacer1, row4_1, row4_spacer2 = st.columns((.7, 7, .7))
            with row4_1:        
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
        if option == "MEILLEURS MODELES" :  
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
        if option == "MODELE RETENU POUR LA MISE EN PRODUCTION" :  
            st.markdown("xxx")
    
    
    ################################################ Demonstration
    elif choice ==  "Démonstration" :
        st.subheader("Analyse de sentiments")  
        with st.form(key='detection_de_sentiment'):
            commentaire = st.text_area("Commentaire client")
            bouton_classification = st.form_submit_button(label="Classifier")

            
        
        if bouton_classification : 
            col1, col2 = st.columns(2)
            
            
            # application des fonction 
            pred_tfidf_lr = prediction_tfidf_lr([commentaire])
            prob_tfidf_lr = prediction_tfidf_lr_proba(commentaire)       
           
            
            with col1:
                    if len(commentaire) == 0:
                        st.warning('This is a warning')
                        st.write("Saisir le commentaire à classifier ")
                    else :    
                        st.success("Commentaire client")
                        st.write(commentaire)
                        st.success("Prediction")
                        if pred_tfidf_lr == 0 :
                            st.write("Negatif 😠")
                        else :
                            st.write("Positif 😃")
                    
                        with col2:
                            st.success("Niveau de confiance")
                            st.write(str(round(prob_tfidf_lr.max(),2) *100) + "%")
        
        st.write(" ")
        st.write(" ")
        
        other_model = st.expander("Comparer le resultat obtenu avec celui d'autres modèle 👉🏼 ")
        with other_model:
            
            options = st.multiselect(
                '',
                ['TF-IDF - Mulitinomial Naive Bayes','Bag of word - Multinomial Naive Bayes','Word2vec - Regression Logistique'])
            
            if 'TF-IDF - Mulitinomial Naive Bayes' in options :
                
                pred_tfidf_MNB = prediction_tfidf_MNB([commentaire])
                prob_tfidf_MNB = prediction_tfidf_MNB_proba(commentaire)
                col1, col2 = st.columns(2)
                with col1:
                    st.write("---")
                    st.success("Prediction TF-IDF _ Mulitinomial Naive Bayes")
                    if pred_tfidf_MNB == 0 :
                        st.write("Negatif 😠")
                    else :
                        st.write("Positif 😃")
                with col2:
                    st.write("---")
                    st.success("Niveau de confiance")
                    st.write(str(round(prob_tfidf_MNB.max(),2) *100) + "%")
                    
            if 'Bag of word - Multinomial Naive Bayes' in options :
                
                pred_BOW_MNB = prediction_BOW_MNB([commentaire])
                prob_BOW_MNB = prediction_BOW_MNB_proba(commentaire)
                col1, col2 = st.columns(2)
                with col1:
                    st.write("---")
                    st.success("Bag of word - Multinomial Naive Bayes")
                    if pred_BOW_MNB == 0 :
                        st.write("Negatif 😠")
                    else :
                        st.write("Positif 😃")
                with col2:
                    st.write("---")
                    st.success("Niveau de confiance")
                    st.write(str(round(prob_BOW_MNB.max(),2) *100) + "%")
                 
                    
                  
                        
                
    row3_spacer1, row3_1, row3_spacer2 = st.columns((.2, 7.1, .2))
    with row3_1:
        st.markdown(footer,unsafe_allow_html=True)#pied de page


if __name__ == '__main__':
  main()                           
            