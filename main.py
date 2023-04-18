import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, BackgroundTasks
import threading

# Creacion de una aplicación FastAPI.

app = FastAPI(title='Consultas Plataformas')

# Cargar el dataframe
df = pd.read_csv('platforms_and_ratings.csv')

#Consultas

@app.get('/get_max_duration/{anio}/{plataforma}/{dtype}')
def get_max_duration(anio: int, plataforma: str, dtype: str):
    df = pd.read_csv("platforms_and_ratings.csv")

    # Filtrar los datos para incluir sólo películas
    df_movies = df[df['type'] == 'movie']
    
    # Filtrar los datos para incluir sólo películas que correspondan al año y la plataforma especificados
    df_movies = df_movies[(df_movies['release_year'] == anio) & (df_movies['platform'] == plataforma)]
    
    # Filtrar los datos para incluir sólo películas con el tipo de duración especificado
    df_movies = df_movies[df_movies['duration_type'] == dtype]
    
    # Ordenar los datos por duración en orden descendente
    df_movies = df_movies.sort_values(by='duration_int', ascending=False)
    
    # Primera fila del dataframe resultante, que tendrá la película con la mayor duración
    max_duration_movie = df_movies.iloc[0]['title']
    
    return {'pelicula': max_duration_movie}


@app.get('/get_score_count/{plataforma}/{scored}/{anio}')
def get_score_count(plataforma: str, scored: float, anio: int):
    df = pd.read_csv("platforms_and_ratings.csv")

    # Filtrar los datos para incluir sólo películas
    df_movies = df[df['type'] == 'movie']
    
    # Filtrar los datos para incluir sólo películas que correspondan al año y la plataforma especificados
    df_movies = df_movies[(df_movies['release_year'] == anio) & (df_movies['platform'] == plataforma)]
    
    # Filtrar los datos para incluir sólo películas con un puntaje mayor que el puntaje especificado
    df_movies = df_movies[df_movies['score_mean'] > scored]
    
    # Contar el número de filas en el dataframe 
    count = len(df_movies)
    
    return {
        'plataforma': plataforma,
        'cantidad': count,
        'anio': anio,
        'score': scored
    }

@app.get('/get_count_platform/{plataforma}')
def get_count_platform(plataforma: str):
    df = pd.read_csv("platforms_and_ratings.csv")

    # Filtrar los datos para incluir sólo películas
    df_movies = df[df['type'] == 'movie']
    
    # Filtrar los datos para incluir sólo películas que correspondan a la plataforma especificada
    if plataforma == 'amazon':
        df_movies = df_movies[df_movies['platform'] == 'amazon']
    elif plataforma == 'netflix':
        df_movies = df_movies[df_movies['platform'] == 'netflix']
    elif plataforma == 'hulu':
        df_movies = df_movies[df_movies['platform'] == 'hulu']
    elif plataforma == 'disney':
        df_movies = df_movies[df_movies['platform'] == 'disney']
    else:
        return {'error': 'Plataforma no reconocida'}
    
    # Contar el número de filas en el dataframe 
    count = len(df_movies)
    
    return {'plataforma': plataforma, 'peliculas': count}


@app.get('/get_actor/{plataforma}/{anio}')
def get_actor(plataforma: str, anio: int):
    df = pd.read_csv("platforms_and_ratings.csv")

   # Filtrar por plataforma y año
    df_filtered = df[(df['platform'] == plataforma) & (df['release_year'] == anio)]

    # Dividir la columna cast en listas de actores, y manejar NaN
    df_filtered.loc[:, 'cast'] = df_filtered['cast'].apply(lambda x: [] if pd.isna(x) else x.split(', '))

    # Crear una lista con todos los actores en el DataFrame
    all_actors = [actor for cast_list in df_filtered['cast'] for actor in cast_list]

    # Obtener el actor que más se repite
    actor_counts = pd.Series(all_actors).value_counts()
    most_common_actor = actor_counts.index[0]
    most_common_actor_appearances = actor_counts[0]

    return {
        'plataforma': plataforma,
        'anio': anio,
        'actor': most_common_actor,
        'apariciones': most_common_actor_appearances
    }

@app.get('/prod_per_county/{tipo}/{pais}/{anio}')
def prod_per_county(tipo: str, pais: str, anio: int):
    df = pd.read_csv("platforms_and_ratings.csv")

    # Filtrar por tipo de contenido, país y año
    df_filt = df[(df['type'] == tipo) & (df['country'] == pais) & (df['release_year'] == anio)]

    # Agrupar por país, año y tipo de contenido
    grouped = df_filt.groupby(['country', 'release_year', 'type'])

    # Contar el número de películas o series por grupo
    count = grouped['title'].count().reset_index(name='count')

    # Obtener el número de películas o series según el tipo especificado
    result = count[count['type'] == tipo]['count'].values[0]
    
    return {'pais': pais, 'anio': anio, tipo: result}


@app.get('/get_contents/{rating}')
def get_contents(rating: str):
    df = pd.read_csv("platforms_and_ratings.csv")

    # Cuenta el número de filas donde el valor de la columna "rating" es igual a la entrada "rating"
    count = len(df[df['rating'] == rating])
    
    return {'rating': rating, 'contenido': count}


@app.get('/get_recommendation/{title}')
def oooget_recommendation(title: str):
    df = pd.read_csv("platforms_and_ratings.csv")

    # Crear una instancia del vectorizador TF-IDF
    vectorizer = CountVectorizer(stop_words="english")

    # Obtener la matriz de términos de los títulos
    title_matrix = vectorizer.fit_transform(df["title"])

    # Calcular la similitud coseno entre todos los títulos
    cosine_similarities = cosine_similarity(title_matrix)

    # Obtener el índice de la película de interés
    idx = df.index[df['title'] == title].tolist()[0]

    # Obtener las puntuaciones de similitud de esa película con todas las películas
    similarity_scores = list(enumerate(cosine_similarities[idx]))

    # Ordenar las películas según puntuación de similitud
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Obtener los índices de las películas recomendadas
    movie_indices = [i[0] for i in similarity_scores[1:11]]

    # Recomendaciones
    recommendations = df.loc[movie_indices, "title"].tolist()[:5]

    # Ejecutar la tarea en segundo plano
    def send():
        pass

    threading.Thread(target=send).start()

    return {'recomendación': recommendations}
