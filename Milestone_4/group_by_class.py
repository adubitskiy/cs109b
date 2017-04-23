import shutil
import time

import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

root_dir = '..'


def load_genre_df():
    start = time.time()
    genre_df = pd.read_csv(root_dir + '/data/genre_df.csv')
    elapsed = time.time() - start
    print("load: %.1f secs" % elapsed)
    return genre_df


def get_image_filename(image_dir, tmdb_id):
    return image_dir + repr(tmdb_id) + '.jpg'


def populate_dir(sample, target_dir, image_dir):
    for tmdb_id in sample:
        image_filename = get_image_filename(image_dir, tmdb_id)
        shutil.copy(image_filename, target_dir)


def populate_data_dir(genre_df):
    image_dir = root_dir + '/posters224/'
    missing_indices = []
    for index, tmdb_id in genre_df['tmdb_id'].iteritems():
        file_name = get_image_filename(image_dir, tmdb_id)
        if not os.path.exists(file_name):
            print('missing:', tmdb_id)
            missing_indices.append(index)

    genre_df.drop(missing_indices, inplace=True)

    num_top_genres = 2
    top_genres = [name for name, count in genre_df['genre'].value_counts()[:num_top_genres].iteritems()]
    print(top_genres)

    num_movies_per_genre = 64
    np.random.seed(109)
    data_dir = 'data'
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)
    os.makedirs('data/train')
    os.makedirs('data/validation')

    genre_groups = genre_df.groupby('genre')
    for name in top_genres:
        group = genre_groups.get_group(name)
        name = name.replace(' ', '_')
        train_dir = 'data/train/' + name + '/'
        validation_dir = 'data/validation/' + name + '/'

        os.makedirs(train_dir)
        os.makedirs(validation_dir)

        sample = np.random.choice(group['tmdb_id'], size=num_movies_per_genre, replace=False)
        train_sample, validation_sample = train_test_split(sample)
        print(len(train_sample), len(validation_sample))
        populate_dir(train_sample, train_dir, image_dir)
        populate_dir(validation_sample, validation_dir, image_dir)


def main():
    genre_df = load_genre_df()

    populate_data_dir(genre_df)


if __name__ == '__main__':
    main()
