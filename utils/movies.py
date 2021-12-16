# fetch movies

# modules
from tmdbv3api import TMDb, Movie, Discover
from imdb import IMDb

import pickle
import pandas as pd

# config
from config import *


def get_list_by_year(start, end):
    """
    Get a list of movie names by year specified.
    """

    tmdb = TMDb()
    tmdb.api_key = api_key
    discover = Discover()

    movietitles = []

    # import by years
    for years in range(start, end):
        # import each page
        for pages in range(1,5):
            results = discover.discover_movies({
                'year': years,
                'page': pages
            })
            for result in results:
                movietitles.append(result.title)

    return movietitles


def search(start_year, end_year, features=features_default, path=path, save_mode='w'):
    """
    Search for all required data in the given list and assemble raw data frame.
    """

    # tmdb setup
    tmdb_setup = TMDb()
    tmdb_setup.api_key = api_key
    tmdb_setup.language = language
    tmdb_setup.debug = debug
    tmdb = Movie()

    # imdb setup
    imdb = IMDb()

    # cache
    infos = []

    # get movie list
    movies = get_list_by_year(start_year, end_year)
    print('Searcing for {} movies...'.format(len(movies)))

    for movie in movies:
        print('Searching for {}...'.format(movie))
        try:
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

        except:
            print('Fetch {} failed'.format(movie))

    # assemble dataframe & drop duplicates
    df = pd.DataFrame(infos)
    df.drop_duplicates('id', inplace=True)

    df.to_csv(path, mode=save_mode, index=False)
    print('Saved to {}'.format(path))

    return path

