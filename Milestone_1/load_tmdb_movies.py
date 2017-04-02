import ast
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd
import sys

reload(sys)
sys.setdefaultencoding('utf8')

TMDB_MOVIES_COLUMN_NAMES = [
    'adult', 'backdrop_path', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'id', 'imdb_id',
    'original_language', 'original_title', 'overview', 'popularity', 'poster_path', 'production_companies',
    'production_countries', 'release_date', 'revenue', 'runtime', 'spoken_languages', 'status', 'tagline', 'title',
    'video', 'vote_average', 'vote_count',
]


def load_tmdb_movies():
    df = pd.read_csv('tmdb_movies_3700.csv', header=None, names=TMDB_MOVIES_COLUMN_NAMES)
    for column_name in ['genres', 'spoken_languages']:
        df[column_name] = df[column_name].map(lambda d: ast.literal_eval(d))
    return df


def get_fig_size(nrows=1):
    return 15, 10 * nrows


def explore_num_genres_per_movie(num_genres_per_movie_list):
    unique_num_genres_per_movie = set(num_genres_per_movie_list)
    print 'unique values of number of genres per movie: %s' % unique_num_genres_per_movie

    num_genres_mean = np.mean(num_genres_per_movie_list)

    _, ax = plt.subplots(1, 1, figsize=get_fig_size())
    ax.hist(num_genres_per_movie_list, bins=np.arange(-0.5, 11.0, step=1.0), alpha=0.4)
    ax.axvline(x=num_genres_mean, linewidth=2, color='k')
    plt.text(num_genres_mean + 0.1, 3300, 'num_genres_mean = %.2f' % num_genres_mean)
    ax.set_xlabel('number of genres per movie')
    ax.set_ylabel('count')
    ax.set_title('Number of genres per movie')

    plt.tight_layout()
    plt.show()


def explore_genre_pairs():
    tmdb_movies_df = load_tmdb_movies()
    num_rows = len(tmdb_movies_df)

    genres_rows = tmdb_movies_df['genres']
    genre_list = []
    num_genres_per_movie_list = []
    for genres in genres_rows:
        num_genres = len(genres)
        num_genres_per_movie_list.append(num_genres)
        for genre in genres:
            genre_list.append(genre['name'])

    explore_num_genres_per_movie(num_genres_per_movie_list)

    print 'avg number of genres per movie: %.3f' % (float(len(genre_list)) / num_rows)

    unique_genres = set(genre_list)
    print 'unique genres: %s' % unique_genres
    print 'number of unique genres: %d' % len(unique_genres)

    counter = Counter(genre_list)
    counter_dict = {k: float(v) / num_rows for k, v in dict(counter).items()}

    sorted_counter_dict = sorted(counter_dict.items(), key=operator.itemgetter(1), reverse=True)
    print sorted_counter_dict


def main():
    explore_genre_pairs()


if __name__ == '__main__':
    main()
