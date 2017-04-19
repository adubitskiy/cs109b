import cPickle

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import make_scorer, hamming_loss, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def load_part(file_name):
    with open(file_name, 'rb') as handle:
        return cPickle.load(handle)


def prepare_data():
    root_folder = '..'

    plot_dict = load_part(root_folder + '/data/plot.pickle')
    # imdb_dict = load_part(root_folder + '/data/cast.pickle')

    # load TMDB
    tmdb_dict = load_part(root_folder + '/data/tmdb_info.pickle')
    columns = [
        'director',
        'cast',
        'casting director',
        'miscellaneous crew',
        'original music',
        'producer',
        'cinematographer',
        'costume designer',
        'art direction',
    ]

    # add 'overview' from TMDB to 'plot' from IMDB (it is a list)
    for tmdb_id, imdb_movie in plot_dict.iteritems():
        if ('plot' in imdb_movie and tmdb_id in tmdb_dict and 'overview' in tmdb_dict[tmdb_id].__dict__ and
                    tmdb_dict[tmdb_id].__dict__['overview'] is not None):
            imdb_movie['plot'].append(tmdb_dict[tmdb_id].__dict__['overview'])

    labels = np.array([d['genres'] for d in plot_dict.values() if 'genres' in d and 'plot' in d])

    mlb = MultiLabelBinarizer()
    label_df = pd.DataFrame(mlb.fit_transform(labels))
    label_df.columns = mlb.classes_
    label_number_df = pd.DataFrame({'cnt': label_df.sum(axis=0)})

    # only leave genres mentioned in 2000 movies or more
    cutoff = 2000
    major_genres = set(label_number_df[label_number_df['cnt'] > cutoff].index)

    # find labels only for the major genres
    labels = np.array(
        [major_genres.intersection(d['genres']) for d in plot_dict.values() if 'genres' in d and 'plot' in d])

    # create the labels vector with only major genres
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)

    # the plot consists of a few parts, join them together
    text_features = np.array([''.join(d['plot']) for d in plot_dict.values() if 'genres' in d and 'plot' in d])

    return text_features, y, mlb.classes_


def get_sample(text_features, y):
    n_elements = len(text_features)
    print n_elements

    np.random.seed(109)
    sample_indices = np.random.choice(n_elements, size=10000, replace=False)

    text_features_sample = text_features[sample_indices]
    y_sample = y[sample_indices]

    return text_features_sample, y_sample


def sgd(X_test, X_train, y_test, y_train, mlb_classes):
    param_grid = {
        'estimator__alpha': np.logspace(-5, -3, num=30),
    }
    model = OneVsRestClassifier(SGDClassifier(class_weight='balanced', random_state=761))
    model_tuning = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=make_scorer(hamming_loss, greater_is_better=False),
        n_jobs=2,
        verbose=1,
    )
    model_tuning.fit(X_train, y_train)
    print model_tuning.best_params_
    print classification_report(y_train, model_tuning.predict(X_train), target_names=mlb_classes)
    print classification_report(y_test, model_tuning.predict(X_test), target_names=mlb_classes)


def random_forest(X_test, X_train, y_test, y_train, mlb_classes):
    param_grid = {
        'min_samples_leaf': (1, 2, 50),
        'max_features': ('auto', 0.2),
    }
    model = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=761)
    model_tuning = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=make_scorer(hamming_loss, greater_is_better=False),
        cv=3,
        n_jobs=2,
        verbose=3,
    )
    model_tuning.fit(X_train, y_train)
    print model_tuning.best_params_
    print classification_report(y_train, model_tuning.predict(X_train), target_names=mlb_classes)
    print classification_report(y_test, model_tuning.predict(X_test), target_names=mlb_classes)


def main():
    text_features, y, mlb_classes = prepare_data()
    text_features_sample, y_sample = get_sample(text_features, y)

    # split into test / train data
    F_train, F_test, y_train, y_test = train_test_split(text_features_sample, y_sample, test_size=0.25, random_state=42)
    # convert into bag of words
    vectorizer = TfidfVectorizer(
        stop_words=stopwords.words("english"),
        token_pattern='[a-zA-Z]+[0-9]*',
        max_df=0.9,
        min_df=0.0001,
        dtype=np.float32,
    )
    X_train = vectorizer.fit_transform(F_train)
    X_test = vectorizer.transform(F_test)
    print 'Train label matrix shape:', y_train.shape
    print 'Train predictor matrix shape:', X_train.shape
    print 'Test label matrix shape:', y_test.shape
    print 'Test predictor matrix shape:', X_test.shape

    # sgd(X_test, X_train, y_test, y_train, mlb_classes)
    random_forest(X_test, X_train, y_test, y_train, mlb_classes)


if __name__ == '__main__':
    main()
