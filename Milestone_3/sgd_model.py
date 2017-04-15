import cPickle
import time
from collections import defaultdict, Counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


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
        'revenue',
        'budget',
        'vote_count',
        'vote_average',
        'cast',
        'crew',

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


def prepare_cast(movie_df, genre_name):
    cast_list = [cast_member['name'] for movie_cast_list in movie_df[genre_name] for cast_member in movie_cast_list]
    cast_counter = Counter(cast_list)

    appearances_limit = 3
    included_cast_list = [cast_name for cast_name, num_movies in cast_counter.iteritems()
                          if num_movies >= appearances_limit]
    included_cast_set = set(included_cast_list)

    num_movies = len(movie_df)
    print num_movies
    movie_attribute_dict = defaultdict(lambda: np.zeros((num_movies,), dtype=np.uint8))

    for i, movie_cast_list in enumerate(movie_df[genre_name]):
        for cast_member in movie_cast_list:
            cast_name = cast_member['name']
            if cast_name in included_cast_set:
                movie_attribute_dict[genre_name + '_' + cast_name][i] = 1

    new_movie_df = movie_df.drop(genre_name, axis=1)

    for key, column in movie_attribute_dict.iteritems():
        new_movie_df[key] = column

    print new_movie_df.shape
    return new_movie_df


def apply_pca(X_train, X_test):
    print "before scaling:"
    standard_scaler = StandardScaler()
    X_train_scaled = standard_scaler.fit_transform(X_train)
    X_test_scaled = standard_scaler.transform(X_test)

    print "before pca"
    pca = PCA()
    pca.fit(X_train_scaled)

    cutoff_index = np.argmin(np.cumsum(pca.explained_variance_ratio_) <= 0.9)

    pca_X_train = pca.transform(X_train_scaled)
    pca_X_test = pca.transform(X_test_scaled)

    pca_X_train = pca_X_train[:, :cutoff_index]
    pca_X_test = pca_X_test[:, :cutoff_index]

    print pca_X_train.shape
    print pca_X_test.shape

    return pca_X_train, pca_X_test


def prepare_movie_df(movie_df):
    reduced_movie_df = get_reduced_movie_df(movie_df)
    reduced_movie_df = prepare_genre_columns(reduced_movie_df)
    reduced_movie_df = prepare_cast(reduced_movie_df, 'cast')
    reduced_movie_df = prepare_cast(reduced_movie_df, 'crew')

    return reduced_movie_df


def default_score(classifier, X, y):
    y_pred = classifier.predict(X)
    return accuracy_score(y, y_pred)


def f1_score_f(classifier, X, y):
    y_pred = classifier.predict(X)
    return f1_score(y, y_pred)


def run_model_with_y_matrix(train_df, test_df, classifier, score_f):
    X_columns = [column for column in train_df.columns if not column.startswith('genre_')]
    y_columns = [column for column in train_df.columns if column.startswith('genre_')]

    X_train = train_df[X_columns]
    X_test = test_df[X_columns]

    X_train, X_test = apply_pca(X_train, X_test)

    y_train = train_df[y_columns]
    y_test = test_df[y_columns]

    classifier.fit(X_train, y_train)

    train_score = score_f(classifier, X_train, y_train)
    test_score = score_f(classifier, X_test, y_test)

    # print classifier.best_params_
    print "train: %.3f, test: %.3f" % (train_score, test_score)


def run_model(train_df, test_df, classifier, score_f):
    X_columns = [column for column in train_df.columns if not column.startswith('genre_')]
    y_columns = [column for column in train_df.columns if column.startswith('genre_')]

    X_train = train_df[X_columns]
    X_test = test_df[X_columns]

    X_train, X_test = apply_pca(X_train, X_test)

    y_train = train_df[y_columns]
    y_test = test_df[y_columns]

    print 'y_train shape:'
    print y_train.shape
    print y_test.shape

    train_score_list = []
    test_score_list = []

    print classifier

    y_train_pred_list = []
    y_test_pred_list = []

    for y_column in y_columns:
        y_train_col = train_df[y_column]
        y_test_col = test_df[y_column]

        train_score, test_score, y_train_pred, y_test_pred = run_model_one_y(y_column, X_train, X_test, y_train_col,
                                                                             y_test_col, classifier, score_f)

        train_score_list.append(train_score)
        test_score_list.append(test_score)

        y_train_pred_list.append(y_train_pred)
        y_test_pred_list.append(y_test_pred)

    y_train_pred_result = np.array(y_train_pred_list).T
    y_test_pred_result = np.array(y_test_pred_list).T

    print 'train'

    print 'y_train'
    print y_train.shape
    print y_train
    print 'y_train_pred_result'
    print y_train_pred_result.shape
    print y_train_pred_result

    print classification_report(y_train, y_train_pred_result)
    print 'test'
    print classification_report(y_test, y_test_pred_result)

    print "train: %.3f, test: %.3f" % (np.mean(train_score_list), np.mean(test_score_list))


def run_model_one_y(genre, X_train, X_test, y_train, y_test, classifier, score_f):
    classifier.fit(X_train, y_train)

    train_score = score_f(classifier, X_train, y_train)
    test_score = score_f(classifier, X_test, y_test)

    print classifier.best_params_
    print "train: %.3f, test: %.3f (%s)" % (train_score, test_score, genre)

    return train_score, test_score, classifier.predict(X_train), classifier.predict(X_test)


def main():
    movie_df = load_movie_df()

    reduced_movie_df = prepare_movie_df(movie_df)
    train_df, test_df = train_test_split(reduced_movie_df, random_state=109)

    # run_model(train_df, test_df, classifier=DummyClassifier(strategy="most_frequent"), score_f=default_score)

    # run_model_2(train_df, test_df, classifier=DummyClassifier(strategy="most_frequent"), score_f=f1_score_f)
    # run_model_2(train_df, test_df, classifier=DummyClassifier(strategy="stratified"), score_f=f1_score_f)
    # run_model_2(train_df, test_df, classifier=DummyClassifier(strategy="uniform"), score_f=f1_score_f)

    # run_model_2(train_df, test_df, classifier=DummyClassifier(strategy="most_frequent"), score_f=default_score)
    # run_model_2(train_df, test_df, classifier=DummyClassifier(strategy="stratified"), score_f=default_score)
    # run_model_2(train_df, test_df, classifier=DummyClassifier(strategy="uniform"), score_f=default_score)

    # run_model(train_df, test_df, classifier=SVC(class_weight='balanced', kernel='linear'), score_f=f1_score_f)

    estimator = GridSearchCV(
        estimator=SGDClassifier(class_weight='balanced'),
        param_grid={
            'alpha': np.logspace(-6, 3, num=10),
        },
        scoring=f1_score_f,
        cv=3,
        verbose=1,
    )

    # estimator = SGDClassifier(class_weight='balanced', alpha=10 ** -2)

    # run_model_2(train_df, test_df, classifier=estimator, score_f=default_score)
    run_model(train_df, test_df, classifier=estimator, score_f=f1_score_f)

    # run_model(train_df, test_df, classifier=SVC(class_weight='balanced', kernel='linear'), score_f=f1_score_f)


if __name__ == '__main__':
    main()
