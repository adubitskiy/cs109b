import cPickle
import pickle
import time
from collections import defaultdict

import numpy as np
import pandas as pd


def load_movie_dict():
    start = time.time()
    with open(r"../data/tmdb_info.pickle", "rb") as input_file:
        movie_dict = cPickle.load(input_file)
    elapsed = time.time() - start
    print "load: %.1f secs" % elapsed
    return movie_dict


def get_movie_attribute_name_list(movie_dict):
    attr_set = set()
    for movie in movie_dict.itervalues():
        attr_set |= set(movie.__dict__.keys())
    return list(attr_set)


def get_movie_df(movie_dict, column_name_list):
    movie_attribute_dict = defaultdict(list)
    for movie in movie_dict.itervalues():
        for movie_attribute_name in column_name_list:
            attr_list = movie_attribute_dict[movie_attribute_name]
            attr_list.append(getattr(movie, movie_attribute_name))

    return pd.DataFrame(movie_attribute_dict)


def save_subsample(movie_dict, sample_size=5000):
    all_keys = list(movie_dict.keys())
    print len(all_keys)

    np.random.seed(109)
    sample_keys = np.random.choice(all_keys, size=sample_size, replace=False)

    sample_movie_dict = {key: movie_dict[key] for key in sample_keys}

    column_name_list = get_movie_attribute_name_list(sample_movie_dict)

    sample_movie_df = get_movie_df(sample_movie_dict, column_name_list)

    print sample_movie_df.shape
    print sample_movie_df

    with open('tmdb_df_%s.pickle' % sample_size, 'wb') as output:
        pickle.dump(sample_movie_df, output, pickle.HIGHEST_PROTOCOL)


def main():
    movie_dict = load_movie_dict()
    save_subsample(movie_dict, sample_size=10000)


if __name__ == '__main__':
    main()
