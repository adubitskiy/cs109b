import pickle
import time

import numpy as np
import pandas as pd
from collections import Counter


def load_movie_dict():
    start = time.time()
    with open(r"../data/tmdb_info.pickle", "rb") as input_file:
        movie_dict = pickle.load(input_file)
    elapsed = time.time() - start
    print("load: %.1f secs" % elapsed)
    return movie_dict


def get_genre_df(movie_dict):
    all_genre_list = []
    for tmdb_id, movie in movie_dict.items():
        all_genre_list.extend(genre['name'] for genre in movie.genres)

    genre_counter = Counter(all_genre_list)
    print('multiple genres:')
    print(genre_counter)

    sorted_genre_name_list = [genre_name for genre_name, _ in genre_counter.most_common()]
    genre_to_index_dict = {genre_name: i for i, genre_name in enumerate(sorted_genre_name_list)}

    tmdb_id_column = []
    genre_column = []
    for tmdb_id, movie in movie_dict.items():
        genre_list = [genre['name'] for genre in movie.genres]
        if genre_list:
            min_index = np.min([genre_to_index_dict[genre_name] for genre_name in genre_list])
            dominant_genre_name = sorted_genre_name_list[min_index]
            tmdb_id_column.append(tmdb_id)
            genre_column.append(dominant_genre_name)

    print(len(tmdb_id_column))
    print('one genre per movie (most dominant):')
    print(Counter(genre_column))

    return pd.DataFrame({'tmdb_id': tmdb_id_column, 'genre': genre_column})


def main():
    movie_dict = load_movie_dict()
    print(len(movie_dict))

    genre_df = get_genre_df(movie_dict)

    genre_df.to_csv('genre_df.csv', index=False, columns=['tmdb_id', 'genre'])


if __name__ == '__main__':
    main()
