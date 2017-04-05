import pandas as pd
import sys
import time
import tmdbsimple as tmdb
from requests import HTTPError

reload(sys)
sys.setdefaultencoding('utf8')


def main():
    start_time = time.time()

    tmdb.API_KEY = 'a995a7fe53e021d77d82b99428850ff1'

    discover = tmdb.Discover()

    start_page = 1
    num_pages = 1000

    for page in xrange(start_page, num_pages + 1):
        if page != start_page:
            time.sleep(7)

        elapsed_mins = (time.time() - start_time) / 60.0

        print 'page: %d, elapsed mins: %.2f' % (page, elapsed_mins)
        discover_result = discover.movie(page=page, sort_by='popularity.desc')

        movies = discover_result['results']

        row_list = []
        for movie in movies:
            movie_id = movie['id']
            print movie_id, movie['title']
            try:
                tmdb_movie = tmdb.Movies(movie_id).info()
                row_list.append(tmdb_movie)
            except HTTPError as e:
                print str(e)

        df = pd.DataFrame(row_list)

        with open('tmdb_movies.csv', 'a') as csv_file:
            df.to_csv(csv_file, header=False, index=False, encoding='utf-8')


if __name__ == '__main__':
    main()
