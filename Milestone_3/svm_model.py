import cPickle
import time
from collections import defaultdict

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def load_movie_df():
    start = time.time()
    with open(r"../data/tmdb_df_5k.pickle", "rb") as input_file:
        movie_df = cPickle.load(input_file)
    elapsed = time.time() - start
    print "load: %.1f secs" % elapsed
    return movie_df


def get_reduced_movie_df(movie_df):
    movie_attribute_name_list = [
        'popularity',
        'genres',
    ]
    return movie_df[movie_attribute_name_list]


def prepare_genre_columns(movie_df):
    num_movies = len(movie_df)
    genre_df_dict = defaultdict(lambda: np.zeros((num_movies,), dtype=np.uint8))

    for i, genre_list in enumerate(movie_df['genres']):
        for genre in genre_list:
            genre_name = genre['name']
            genre_df_dict['genre_' + genre_name][i] = 1

    new_movie_df = movie_df.drop("genres", axis=1)

    for key, column in genre_df_dict.iteritems():
        new_movie_df[key] = column

    return new_movie_df


def prepare_movie_df(movie_df):
    reduced_movie_df = get_reduced_movie_df(movie_df)
    reduced_movie_df = prepare_genre_columns(reduced_movie_df)
    return reduced_movie_df


def run_model_one_y(genre, X_train, X_test, y_train, y_test, classifier):
    classifier.fit(X_train, y_train)

    train_score = classifier.score(X_train, y_train)
    test_score = classifier.score(X_test, y_test)

    print "train: %.3f, test: %.3f (%s)" % (train_score, test_score, genre)

    return train_score, test_score


def run_model(train_df, test_df, classifier):
    X_columns = [column for column in train_df.columns if not column.startswith('genre_')]
    y_columns = [column for column in train_df.columns if column.startswith('genre_')]

    X_train = train_df[X_columns]
    X_test = test_df[X_columns]

    train_score_list = []
    test_score_list = []

    print classifier

    for y_column in y_columns:
        y_train = train_df[y_column]
        y_test = test_df[y_column]

        train_score, test_score = run_model_one_y(y_column, X_train, X_test, y_train, y_test, classifier)
        train_score_list.append(train_score)
        test_score_list.append(test_score)

    print "train: %.3f, test: %.3f" % (np.mean(train_score_list), np.mean(test_score_list))


def main():
    movie_df = load_movie_df()

    reduced_movie_df = prepare_movie_df(movie_df)
    train_df, test_df = train_test_split(reduced_movie_df, random_state=109)

    run_model(train_df, test_df, classifier=DummyClassifier(strategy="most_frequent"))
    run_model(train_df, test_df, classifier=SVC())


if __name__ == '__main__':
    main()
