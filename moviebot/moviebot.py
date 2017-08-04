#Movie recommendation code adapted from https://beckernick.github.io/matrix-factorization-recommender/
from flask import Flask
from flask_ask import Ask, statement, question
import requests
import json
import numpy as np
from scipy.sparse.linalg import svds
import pandas as pd

movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
links = pd.read_csv('ml-latest-small/links.csv')

app = Flask(__name__)
ask = Ask(app, '/')

key = '397922525c457123a2c27b7125e7868e' #PLEASE DO NOT ABUSE
idx = 0
@app.route('/')
def homepage():
    return "Hello"

@ask.launch
def start_skill():
    msg = "Lets talk about movies!"
    return question(msg)

def get_id(query):
    """
    Takes a movie name (string) and returns the tmdb id
    """
    print(query)
    link = "https://api.themoviedb.org/3/search/movie?api_key={}&language=en-US&query='{}'&page=1&include_adult=false".format(key, query)
    response = requests.get(link)
    j = json.loads(response.content)['results'][0]
    return j['id']

def get_info(movie_id):
    """
    Takes a tmdbId id (int) and returns a dictionary of information
    """
    link = "https://api.themoviedb.org/3/movie/{}?api_key={}&language=en-US".format(movie_id, key)
    response = requests.get(link)
    j = json.loads(response.content)
    return j

@ask.intent("OverviewIntent")
def OverviewIntent(movie):
    """
    Takes a movie name (string) and returns the tmdb overview
    """
    info = get_info(get_id(movie))
    return statement(info['overview'])

@ask.intent("GenreIntent")
def GenreIntent(movie):
    """
    Takes a movie name (string) and returns the genre
    """
    info = get_info(get_id(movie))
    msg = '{} is a '.format(movie)
    for i, g in enumerate(info['genres']):
        msg += g['name'] + ' '
        if i != len(info['genres']) - 1:
            msg += ', '
        if i == len(info['genres']) - 2:
            msg += 'and '
    msg += 'movie'
    return statement(msg)

@ask.intent("RatingIntent")
def RatingIntent(movie):
    """
    Takes a movie name (string) and returns the tmdb rating
    """
    info = get_info(get_id(movie))
    return statement('It has a {} rating'.format(info['vote_average']))


@ask.intent("DateIntent")
def DateIntent(movie):
    """
    Takes a movie name (string) and returns the release date
    """
    info = get_info(get_id(movie))
    return statement(info['release_date'])

def recommend_movies(predictions_df, userId, movies_df, original_ratings_df, num_recommendations=5):

    # Get and sort the user's predictions
    user_row_number = userId - 1 # userId starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.userId == (userId)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').
                     sort_values(['rating'], ascending=False)
                 )

    print('User {0} has already rated {1} movies.'.format(userId, user_full.shape[0]))
    print('Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations))

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieId',
               right_on = 'movieId').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations
def getMovietmdbId(movieId):
    """
    Takes a movie id and returns the tmdb id
    """
    try:
        m = links.loc[links['movieId'] == movieId].as_matrix()[0]
        return int(m[2])
    except IndexError:
        print('Could not find a movie')

def getRec(tmdbIds, num=5):
    """
    Takes a list of tmdb ids and returns (num) recommendations
    """
    global ratings
    l = []
    id = int(max(ratings['userId']) + 1)
    for i in tmdbIds:
        print(i)
        l = {'userId':id, 'movieId': getMovietmdbId(i), 'rating': 5, 'timestamp': 0}
        ratings = ratings.append(l, ignore_index=True)
    print(ratings.tail(), id)
    R_df = ratings.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
    R = R_df.as_matrix()
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(R_demeaned, k = 50)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)
    already_rated, predictions = recommend_movies(preds_df, id, movies, ratings, num)
    return already_rated, predictions

ids = []

@ask.intent("RecommendationIntent")
def RecommendationIntent():
    """
    Checks if user has given liked movies. If so, return predictions.
    Else ask for liked movies
    """
    global predictions
    global ids
    #global pidx
    #pidx = 0
    if len(ids) == 0:
        return question('What movie do you like?')
    else:
        already_rated, predictions = getRec(ids)
        msg = 'I recommend '
        for i in range(len(predictions)):
            msg += get_info(getMovietmdbId(predictions.iloc[i]['movieId']))['title'] + ', '
        #print(msg)
        return statement(msg)

@ask.intent("LikeIntent")
def likeIntent(movie):
    """
    Takes a movie name (string) and adds it to users liked movies
    """
    global ids
    ids.append(get_id(movie))
    return question('What else?')

if __name__ == '__main__':
    app.run(debug=True)
