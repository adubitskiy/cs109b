import cPickle

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, zero_one_loss, \
    jaccard_similarity_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

root_folder = '..'


def load_part(file_name):
    with open(file_name, 'rb') as handle:
        return cPickle.load(handle)


def cutoff_labels(labels, cutoff):
    mlb = MultiLabelBinarizer()
    label_df = pd.DataFrame(mlb.fit_transform(labels))
    label_df.columns = mlb.classes_
    label_number_df = pd.DataFrame({'cnt': label_df.sum(axis=0)})
    major_genres = set(label_number_df[label_number_df['cnt'] > cutoff].index)
    return major_genres


def get_major_genres(plot_dict):
    labels = np.array([d['genres'] for d in plot_dict.values() if 'genres' in d and 'plot' in d])
    # only leave genres mentioned in 2000 movies or more
    major_genres = cutoff_labels(labels, 2000)
    return major_genres


def prepare_cast_data(tmdb_dict, cast_dict, major_genres, sample_tmdb_ids):
    columns = [
        'director',
        'cast',
        'casting director',
        'miscellaneous crew',
        'original music',
        'producer',
        'cinematographer',
        'costume designer',
        'art direction']

    labels = np.array(
        [major_genres.intersection(cast_dict[tmdb_id]['genres']) for tmdb_id in sample_tmdb_ids if
         'genres' in cast_dict[tmdb_id]])
    # create the labels vector with only major genres
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)

    # combine all names separated by '|'
    features = []
    for tmdb_id in sample_tmdb_ids:
        imdb_movie = cast_dict[tmdb_id]
        if 'genres' not in imdb_movie:
            continue
        l = []
        for c in columns:
            if (c in imdb_movie):
                l = l + [c['name'].encode('utf-8') for c in imdb_movie[c]]
        # add crew and cast from TMDB
        if (tmdb_id in tmdb_dict):
            tmdb_movie = tmdb_dict[tmdb_id].__dict__
            if ('crew' in tmdb_movie):
                l = l + [c['name'].encode('utf-8') for c in tmdb_movie['crew']]
            if ('cast' in tmdb_movie):
                l = l + [c['name'].encode('utf-8') for c in tmdb_movie['cast']]
        # remove duplicates before joiniing
        features.append('|'.join(set(l)))

    vectorizer = CountVectorizer(
        max_df=0.99,
        min_df=0.0002,
        stop_words=stopwords.words("english"),
        tokenizer=lambda x: x.split('|'),
        dtype=np.float32)

    return features, y, mlb.classes_, vectorizer


def prepare_text_data(tmdb_dict, plot_dict, major_genres, sample_tmdb_ids):
    # add 'overview' from TMDB to 'plot' from IMDB (it is a list)
    for tmdb_id, imdb_movie in plot_dict.iteritems():
        if ('plot' in imdb_movie and tmdb_id in tmdb_dict and 'overview' in tmdb_dict[tmdb_id].__dict__ and
                    tmdb_dict[tmdb_id].__dict__['overview'] is not None):
            imdb_movie['plot'].append(tmdb_dict[tmdb_id].__dict__['overview'])

    labels = np.array(
        [major_genres.intersection(plot_dict[tmdb_id]['genres']) for tmdb_id in sample_tmdb_ids if
         'genres' in plot_dict[tmdb_id] and 'plot' in plot_dict[tmdb_id]])
    print len(labels)

    # create the labels vector with only major genres
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)
    # the plot consists of a few parts, join them together
    features = np.array([''.join(plot_dict[tmdb_id]['plot']) for tmdb_id in sample_tmdb_ids if
                         'genres' in plot_dict[tmdb_id] and 'plot' in plot_dict[tmdb_id]])

    vectorizer = TfidfVectorizer(
        stop_words=stopwords.words("english"),
        token_pattern='[a-zA-Z]+[0-9]*',
        max_df=0.9,
        min_df=0.0001,
        dtype=np.float32,
    )
    return features, y, mlb.classes_, vectorizer


def evaluate_baseline(X_train, X_test, y_train, y_test, mlb_classes, strategy):
    print strategy

    num_y_columns = y_train.shape[1]

    y_test_pred_list = []
    for i in xrange(num_y_columns):
        one_y_train = y_train[:, i]
        model = DummyClassifier(strategy=strategy)
        model.fit(X_train, one_y_train)
        one_y_test_pred = model.predict(X_test)
        y_test_pred_list.append(one_y_test_pred)

    y_test_pred = np.array(y_test_pred_list).T

    print 'accuracy score: %.3f' % accuracy_score(y_test, y_test_pred)
    print 'jaccard similarity score: %.3f' % jaccard_similarity_score(y_test, y_test_pred)
    print 'hamming loss: %.3f' % hamming_loss(y_test, y_test_pred)
    print 'zero one loss: %.3f' % zero_one_loss(y_test, y_test_pred)
    print classification_report(y_test, y_test_pred, target_names=mlb_classes)


def f1_score_f(classifier, X, y):
    y_pred = classifier.predict(X)
    return f1_score(y, y_pred, average='micro')


def sgd(X_train, X_test, y_train, y_test, mlb_classes):
    param_grid = {
        'estimator__alpha': np.logspace(-5, -1, num=50),
    }
    model = OneVsRestClassifier(SGDClassifier(class_weight='balanced', random_state=761))
    model_tuning = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=f1_score_f,
        cv=3,
        n_jobs=2,
        verbose=1,
    )
    model_tuning.fit(X_train, y_train)
    y_test_pred = model_tuning.predict(X_test)

    print model_tuning.best_params_
    print 'hamming loss: %.3f' % hamming_loss(y_test, y_test_pred)
    print classification_report(y_train, model_tuning.predict(X_train), target_names=mlb_classes, digits=3)
    print classification_report(y_test, y_test_pred, target_names=mlb_classes, digits=3)

    return (classification_report(y_train, model_tuning.predict(X_train), target_names=mlb_classes),
            classification_report(y_test, model_tuning.predict(X_test), target_names=mlb_classes))


def random_forest(X_test, X_train, y_test, y_train, mlb_classes):
    param_grid = {
        'min_samples_leaf': (1, 2, 5),
        'max_features': ('auto', 0.1, 0.2),
    }
    # model = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=761)
    model = RandomForestClassifier(n_estimators=30, random_state=761, class_weight='balanced')
    # model = RandomForestClassifier(n_estimators=30, random_state=761, max_features=0.2, min_samples_leaf=0.01,
    #                                verbose=3, class_weight='balanced', n_jobs=2)
    model_tuning = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=f1_score_f,
        cv=3,
        n_jobs=2,
        verbose=3,
    )
    model_tuning.fit(X_train, y_train)
    y_test_pred = model_tuning.predict(X_test)

    print model_tuning.best_params_
    print 'hamming loss: %.3f' % hamming_loss(y_test, y_test_pred)
    print classification_report(y_train, model_tuning.predict(X_train), target_names=mlb_classes)
    print classification_report(y_test, y_test_pred, target_names=mlb_classes)

    return (classification_report(y_train, model_tuning.predict(X_train), target_names=mlb_classes),
            classification_report(y_test, model_tuning.predict(X_test), target_names=mlb_classes))


def main():
    # load TMDB movies dataset
    tmdb_movies = load_part(root_folder + '/data/tmdb_info.pickle')

    plot_dict = load_part(root_folder + '/data/plot.pickle')

    cast_dict = load_part(root_folder + '/data/cast10K.pickle')

    major_genres = get_major_genres(plot_dict)

    sample_tmdb_ids = np.array([tmdb_id for tmdb_id, d in cast_dict.items() if
                                'genres' in d and 'genres' in plot_dict[tmdb_id] and tmdb_id in plot_dict and 'plot' in
                                plot_dict[tmdb_id]])
    print len(sample_tmdb_ids)

    # get labels / features from the cast / crew data
    cast_features, cast_y, cast_mlb_classes, cast_vectorizer = prepare_cast_data(tmdb_movies, cast_dict, major_genres,
                                                                                 sample_tmdb_ids)

    print np.shape(cast_features)
    print np.shape(cast_y)
    print len(cast_mlb_classes)

    # get labels / features from the text data
    text_features, text_y, text_mlb_classes, text_vectorizer = prepare_text_data(tmdb_movies, plot_dict, major_genres,
                                                                                 sample_tmdb_ids)

    print np.shape(text_features)
    print np.shape(text_y)
    print len(text_mlb_classes)

    # split into test / train data
    cast_F_train, cast_F_test, text_F_train, text_F_test, y_train, y_test = train_test_split(cast_features,
                                                                                             text_features, cast_y,
                                                                                             test_size=0.25,
                                                                                             random_state=42)

    cast_X_train = cast_vectorizer.fit_transform(cast_F_train)
    cast_X_test = cast_vectorizer.transform(cast_F_test)

    print np.shape(cast_X_train)
    print np.shape(cast_X_test)

    text_X_train = text_vectorizer.fit_transform(text_F_train)
    text_X_test = text_vectorizer.transform(text_F_test)

    print np.shape(text_X_train)
    print np.shape(text_X_test)

    X_train = sparse.hstack((cast_X_train, text_X_train))
    X_test = sparse.hstack((cast_X_test, text_X_test))

    print np.shape(X_train)
    print np.shape(X_test)

    # evaluate_baseline(X_train, X_test, y_train, y_test, cast_mlb_classes, 'stratified')
    # evaluate_baseline(X_train, X_test, y_train, y_test, cast_mlb_classes, 'uniform')
    # evaluate_baseline(X_train, X_test, y_train, y_test, cast_mlb_classes, 'most_frequent')

    # sgd(X_train, X_test, y_train, y_test, cast_mlb_classes)
    random_forest(X_test, X_train, y_test, y_train, cast_mlb_classes)


if __name__ == '__main__':
    main()
