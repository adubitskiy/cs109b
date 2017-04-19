import cPickle

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
    features, y, mlb_classes, vectorizer = prepare_cast_data(tmdb_movies, cast_dict, major_genres, sample_tmdb_ids)

    print np.shape(features)
    print np.shape(y)
    print len(mlb_classes)

    # get labels / features from the text data
    features, y, mlb_classes, vectorizer = prepare_text_data(tmdb_movies, plot_dict, major_genres, sample_tmdb_ids)

    print np.shape(features)
    print np.shape(y)
    print len(mlb_classes)


if __name__ == '__main__':
    main()
