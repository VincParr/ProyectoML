#IMPORT LIBRERIE
import datetime
import numpy as np
import pandas as pd # https://pandas.pydata.org/docs/
import plotly_express as px # https://plotly.com/python-api-reference/index.html
import streamlit as st # https://docs.streamlit.io/
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import ast
import requests
from collections import Counter

from nltk.corpus.reader import reviews
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
import imdb
from bs4 import BeautifulSoup
from sqlalchemy import true


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=1ea4ee4ed14ad984228d22cccaa361df&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    if poster_path is not None:
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    else:
        full_path="Poster extraviado"
    return full_path
    
#SETTO NOME PAGINA
st.set_page_config(
    page_title="Film",
    page_icon=":film_frames:",
    layout="wide"
)

#CARICO IL DATASET
local_filename = 'finale.csv'

def load_dataset():
    df = pd.read_csv(local_filename)
    return df

df = load_dataset() #CARICO IL DATASET

#Elimino colonna, dato che non serve
df.drop({'Unnamed: 0.9'},axis=1,inplace=True)
df.drop({'Unnamed: 0.8'},axis=1,inplace=True)
df.drop({'Unnamed: 0.7'},axis=1,inplace=True)
df.drop({'Unnamed: 0.6'},axis=1,inplace=True)
df.drop({'Unnamed: 0.5'},axis=1,inplace=True)
df.drop({'Unnamed: 0.4'},axis=1,inplace=True)
df.drop({'Unnamed: 0.3'},axis=1,inplace=True)
df.drop({'Unnamed: 0.2'},axis=1,inplace=True)
df.drop({'Unnamed: 0.1'},axis=1,inplace=True)
df.drop({'Unnamed: 0'},axis=1,inplace=True)
df.drop({'popularity'},axis=1,inplace=True)
df.drop({'keywords'},axis=1,inplace=True)
df.drop({'runtime'},axis=1,inplace=True)
df.drop({'spoken_languages'},axis=1,inplace=True)
df.drop({'status'},axis=1,inplace=True)
df.drop({'crew'},axis=1,inplace=True)

#funzione per swappare colonne
def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df
#swappo colonne
df = swap_columns(df, 'title', 'budget')
df = swap_columns(df, 'overview', 'genres')
df = swap_columns(df, 'Director', 'id')
df = swap_columns(df, 'original_language', 'cast')
df = swap_columns(df, 'vote_average', 'original_title')

#Elimino le righe che hanno almeno una colonna vuota
df = df.dropna(axis=0,how='any')
#Trasformo la colonna budget in float e scalo
#df.budget = df.budget.str.replace(',', '').astype(float)
#df.budget = df.budget/1_000_000


year=['2000']*4776
df.insert(6,'year',year)
df['release_date'] =  pd.to_datetime(df['release_date'],format='%Y/%m/%d')
df['year'] = df['release_date'].dt.strftime('%Y')
df.year = df.year.astype(int)
#Poster=['a']*4776
#df.insert(0,'Poster',Poster)
#print(Poster)
#df['Poster'][0]=fetch_poster(df['id'][0])
#df2=df
#for ind in df.index:
    #print(df['id'][ind])
    #df['Poster'][ind]=fetch_poster(df2['id'][ind])
    #print(ind)
    #print(": ")
    #print(fetch_poster(df['id'][ind]))
#print(array_locandina)
#Trasformo la colonna revenue in float e scalo
#df.revenue = df.revenue.str.replace(',', '').astype(float)
df.revenue = df.revenue/1_000_000

#Trasformo la colonna vote_average in float e scalo
#df.vote_average = df.vote_average.astype(str).str.replace(',', '').astype(float)
#df.vote_average = df.vote_average/1_000_000

#Trasformo la colonna vote_count in float e scalo
#df.vote_count = df.vote_count.str.replace(',', '').astype(float)
#df.vote_count = df.vote_count/1_000_000
df.info()
#SIDEBAR
st.sidebar.header("Filtros")

df_filtered = df

st.title("BIENVENIDO A SU TABLERO 🎬")

#FILTRO FILM
film_name = st.sidebar.text_input(
    "Nombre de pelicula",
)
#Filtra i film, lascia quelli che contengono il testo inserito nel Titolo
df_filtered = df_filtered[df_filtered['title'].str.contains(film_name,case=False,na=False)]

#FILTRO ATTORI
director_filter = st.sidebar.text_input(
    "Director",
)

cols_dir = ['Director']
mask_star = df_filtered[cols_dir].apply(
    lambda col:col.str.contains(director_filter,na=False,case=False)
    ).any(axis=1) #per quella riga dai True, se almeno uno dei 4 filtri e' True
    #genera una colonna di True\False che poi vengono usate come maschera dentro df_filtered
df_filtered = df_filtered[mask_star]

####FILTRO MIN ANNO
from datetime import datetime
#df['year'] =  pd.to_datetime(df['year'],format='%Y')
#df['year'] = df['release_date'].dt.strftime('%Y')
#df['year'] =  pd.to_datetime(df['year'],format='%Y')
#df.year = df.year.astype(int)
df.info()
start_year = st.sidebar.slider('Año', min_value=min(df['year']), max_value=max(df['year']))
df_filtered = df_filtered[df_filtered['year']>=start_year]

#####FILTRO MIN RATING
min_vote = st.sidebar.slider('Voto', min_value=min(df['vote_average']), max_value=max(df['vote_average']))
df_filtered = df_filtered[df_filtered['vote_average']>=min_vote]

#####FILTRO MIN GROSS
min_rev = st.sidebar.slider('Ingresos', min_value=min(df['revenue']), max_value=max(df['revenue']))
df_filtered = df_filtered[df_filtered['revenue']>=min_rev]


###MOSTRA KPI
l_col,m_col = st.columns(2)
avgIM = avgGRO = "Assente"
if(len(df_filtered)>0):
    #Calcola medie
    avgIM = f"{np.mean(df_filtered['vote_average']):.2f}"
    avgGRO = f"{np.mean(df_filtered['revenue']):.2f}"

l_col.markdown(f"## Valoración media: {avgIM}")
m_col.markdown(f"## Ingresos medios: {avgGRO}")
#r_col.markdown(f"## Average gross (mils): {avgGRO}")

###Visualizza immagine copertina film
def path_to_image_html(path):
    return '<img src="' + path + '" width="140" >'

def convert_df(input_df):
    return input_df.to_html(escape=False, render_links=True, formatters=dict(Poster=path_to_image_html))

###Mostra tabella
html = convert_df(df_filtered)

st.markdown( f"<div style=\"height: 80vh; overflow: auto\"> {html} </div>",
    unsafe_allow_html=True
)


st.title('Sistema de recomendación')


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movie_posters = []

    for i in movies_list:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))
    return recommended_movies,recommended_movie_posters

movies_dict = pickle.load(open('movie_recm.pkl','rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl','rb'))

nltk.download('stopwords')
# Create your views here
stopset = set(stopwords.words('english'))

#with open('vector.pkl', 'rb') as efile:
   # vectorizer = pickle.load(efile)
#model = joblib.load('model.joblib')
#model = pickle.load(open('svc.pkl','rb'))
#print(model)
def sentiment(movie):
    global reviews
    reviews= []
    sentiments={}
    x=[]
    ia = imdb.IMDb()
    search = ia.search_movie(movie)
    id = search[0].movieID
    page = requests.get('https://www.imdb.com/title/tt{}/reviews?ref_=tt_urv'.format(id))
    soup = BeautifulSoup(page.content, 'html.parser')
    movie_data=soup.find_all('div',attrs= {'class': 'lister-item-content'})
    
    vector1 = pickle.load(open('vector.pkl','rb'))
    model1 = pickle.load(open('modelloSVC2.pkl','rb'))
    
    for store in movie_data:
        review = store.find('a', class_ = 'title').text.replace('\n', '')
        reviews.append(review)
    for i in reviews:
        #movie_vector=vectorizer.transform([i])
        #pred = model.predict(movie_vector)
        #pred=model.predict(vectorizer.transform([i]))
        movie_vector=vector1.transform([i])
        pred = model1.predict(movie_vector)
        print(i, pred)
        print(pred)
        if pred[0]=='0':
            pred="👎😭"
        else:
            pred="👍✅"
        sentiments[i]=pred
        
    return sentiments
    

selected_movie_name = st.selectbox(
    "Elige una pelicula 🍿",
    movies['title'].values
)
if st.button('Mostrar recomendación'):
    names,posters = recommend(selected_movie_name)
    rece=sentiment(selected_movie_name)
    #print(ciao)
    #display with the columns
    col1, col2, col3, col4, col5= st.columns(5)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])
    
    
    st.write('Las opiniones de otros usuarios:')
    sium = pd.DataFrame({
       'Opiniones': sentiment(selected_movie_name)
    })
    st.table(sium)
