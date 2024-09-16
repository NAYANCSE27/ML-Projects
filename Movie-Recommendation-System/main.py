# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:59:57 2024

@author: user
"""

import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import pickle


# datasets input
movies_path = "D:\\ML-Projects\\Movie-Recommendation-System\\tmdb_5000_movies.csv"
credits_path = "D:\\ML-Projects\\Movie-Recommendation-System\\tmdb_5000_credits.csv"

movies = pd.read_csv(movies_path)
credits = pd.read_csv(credits_path)

# print(movies.info())
# print(movies.head(1)['genres'].values)


# merge datasets
movies = pd.merge(movies, credits, on='title')
#print(movies.info())


# datasets preprocessing
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
# print(movies.info())
# print(movies.isnull().sum())
movies.dropna(inplace=True)
# print(movies.isnull().sum())
# print(movies.duplicated().sum())


# apply function for formatting the datasets
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(obj):
    counter = 0
    L = []
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break 
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x:x.split())
# print(movies.info())
# print(movies['overview'])

# removing space in between two words
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
#print(movies['genres'])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])


# creating new columns for movie datasets
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
#print(movies.info())


# creating new dataframe
new_dataframe = movies[['movie_id', 'title', 'tags']]
# print(new_dataframe)
new_dataframe.loc[:, 'tags'] = new_dataframe['tags'].apply(lambda x: " ".join(x))
# print(new_dataframe['tags'])

new_dataframe.loc[:, 'tags'] = new_dataframe['tags'].apply(lambda x: x.lower())
# print(new_dataframe['tags'])


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)

new_dataframe.loc[:, 'tags'] = new_dataframe['tags'].apply(stem)


cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_dataframe['tags']).toarray()
# print(vectors)
# print(cv.get_feature_names_out())


similarity = cosine_similarity(vectors)
# print(similarity)



def recommend(movie):
    movie_index = new_dataframe[new_dataframe['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_dataframe.iloc[i[0]].title)
    
recommend('Batman')
        

pickle.dump(new_dataframe, open('movies.pkl', 'wb'))
pickle.dump(new_dataframe.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))






