import cPickle
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA


def load_movie_dict():
    start = time.time()
    with open(r"tmdb_info.pickle", "rb") as input_file:
        movie_dict = cPickle.load(input_file)
    elapsed = time.time() - start
    print "load: %.1f secs" % elapsed
    return movie_dict


def get_movie_attribute_name_list(movie_dict):
    attr_set = set()
    for movie in movie_dict.itervalues():
        attr_set |= set(movie.__dict__.keys())
    return list(attr_set)


def get_movie_df(movie_dict):
    all_movie_attribute_name_list = [
        # 'poster_path', 'backdrop_path', 'base_uri', 'id', 'imdb_id',
        # 'genres',

        'production_countries', 'overview', 'video',
        'title', 'tagline', 'crew', 'homepage',
        'belongs_to_collection', 'original_language', 'status', 'spoken_languages',
        'adult', 'production_companies',
        'original_title', 'cast',

        'revenue', 'vote_count', 'release_date', 'popularity', 'budget', 'vote_average', 'runtime',
    ]
    movie_attribute_name_list = [
        'revenue',
        'vote_count',
        'popularity',
        'budget',
        'vote_average',
        # 'release_date',
        # 'runtime',
    ]

    movie_attribute_dict = defaultdict(list)
    for movie in movie_dict.itervalues():
        for movie_attribute_name in movie_attribute_name_list:
            attr_list = movie_attribute_dict[movie_attribute_name]
            attr_list.append(getattr(movie, movie_attribute_name))

    return pd.DataFrame(movie_attribute_dict)


def explore_pca(movie_df):
    print movie_df.describe()

    scaled_movies = preprocessing.scale(movie_df)

    pca = PCA(n_components=2)
    pca_X = pca.fit_transform(scaled_movies)

    print "explained variance ratio:"
    print pca.explained_variance_ratio_

    plt.figure(figsize=(10, 8))
    plt.scatter(pca_X[:, 0], pca_X[:, 1])
    plt.show()


def main():
    movie_dict = load_movie_dict()
    movie_df = get_movie_df(movie_dict)
    explore_pca(movie_df)


if __name__ == '__main__':
    main()
