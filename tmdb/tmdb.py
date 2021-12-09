from tmdbv3api import TMDb, Movie

tmdb = TMDb()
tmdb.api_key = '94b42385a681053cab08a06553dcfa19'
tmdb.language = 'en'
tmdb.debug = True

movie = Movie()
search = movie.search('Red Notice')

m = movie.details(search[0]['id'])

