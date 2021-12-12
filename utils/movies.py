# fetch movies

from tmdbv3api import TMDb, Movie
from imdb import IMDb

import pickle
import pandas as pd

from config import *


def search(movies, features=features_default, path=path, save_mode='w'):
    tmdb_setup = TMDb()
    tmdb_setup.api_key = api_key
    tmdb_setup.language = language
    tmdb_setup.debug = debug

    tmdb = Movie()
    imdb = IMDb()
    infos = []

    for movie in movies:
        print('Searching for {}...'.format(movie))
        res = tmdb.search(movie)
        t_id = res[0]['id']
        details = tmdb.details(t_id)
        info = {
            k: details[k] for k in features
        }
        t_rev = tmdb.reviews(t_id)
        info['reviews_tmdb'] = pickle.dumps(t_rev)
        
        i_mov = imdb.search_movie(movie)[0]
        i_rev = imdb.get_movie(i_mov.movieID, ['reviews'])['reviews']
        info['reviews_imdb'] = pickle.dumps(i_rev)

        genres = info['genres']
        info['genres'] = pickle.dumps(genres)

        infos.append(info)

    df = pd.DataFrame(infos)
    
    df.to_csv(path, mode=save_mode, index=False)
    print('Saved to {}'.format(path))

    return path

