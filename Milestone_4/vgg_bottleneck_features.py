import numpy as np
from keras import applications, optimizers
from keras import backend as K
from keras.engine import Model
from keras.layers import Flatten, Dense, Dropout, Conv2D, Activation, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report

top_model_weights_path = 'bottleneck_fc_model.h5'
nb_train_samples = 96
nb_validation_samples = 32
epochs = 50
batch_size = 32
img_width, img_height = 224, 224
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'


def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)
    model = applications.VGG16(include_top=False, weights='imagenet')

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )
    bottleneck_features_train = model.predict_generator(
        train_generator,
        steps=nb_train_samples // batch_size,
        verbose=1,
    )
    np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
    np.save(open('train_classes.npy', 'wb'), train_generator.classes)

    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )
    bottleneck_features_validation = model.predict_generator(
        validation_generator,
        steps=nb_validation_samples // batch_size,
        verbose=1,
    )
    np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)
    np.save(open('validation_classes.npy', 'wb'), validation_generator.classes)


def print_report(train_classes, validation_classes, y_train_pred, y_validation_pred):
    print(train_classes)
    print(y_train_pred)
    print(classification_report(train_classes, y_train_pred))
    print(validation_classes)
    print(y_validation_pred)
    print(classification_report(validation_classes, y_validation_pred))


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_classes = np.load(open('train_classes.npy', 'rb'))
    num_classes = len(np.unique(train_classes))
    print(num_classes)
    train_labels = to_categorical(train_classes)

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_classes = np.load(open('validation_classes.npy', 'rb'))
    validation_labels = to_categorical(validation_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.75))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.75))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        train_data, train_labels,
        epochs=250,
        batch_size=batch_size,
        validation_data=(validation_data, validation_labels),
        verbose=1,
    )

    y_train_pred = model.predict_classes(
        train_data,
        batch_size=batch_size,
        verbose=1,
    )

    y_validation_pred = model.predict_classes(
        validation_data,
        batch_size=batch_size,
        verbose=1,
    )

    print_report(train_classes, validation_classes, y_train_pred, y_validation_pred)

    model.save_weights(top_model_weights_path)


def generator_print_report(model, train_classes, train_generator, validation_classes, validation_generator):
    y_train_proba = model.predict_generator(
        train_generator,
        steps=nb_train_samples // batch_size,
        verbose=1,
    )
    y_train_pred = [np.argmax(y) for y in y_train_proba]
    y_validation_proba = model.predict_generator(
        validation_generator,
        steps=nb_validation_samples // batch_size,
        verbose=1,
    )
    y_validation_pred = [np.argmax(y) for y in y_validation_proba]
    print_report(train_classes, validation_classes, y_train_pred, y_validation_pred)


def fine_tune_convolution_block():
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    print('Model loaded.')

    train_classes = np.load(open('train_classes.npy', 'rb'))
    num_classes = len(np.unique(train_classes))
    print(num_classes)

    validation_classes = np.load(open('validation_classes.npy', 'rb'))

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.75))
    top_model.add(Dense(64, activation='relu'))
    top_model.add(Dropout(0.75))
    top_model.add(Dense(num_classes, activation='softmax'))

    top_model.load_weights(top_model_weights_path)

    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    for layer in model.layers[:15]:
        layer.trainable = False

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy'],
    )

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
    )

    model.summary()

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        verbose=2,
    )

    generator_print_report(model, train_classes, train_generator, validation_classes, validation_generator)


def small_conv_net_from_scratch():
    train_classes = np.load(open('train_classes.npy', 'rb'))
    num_classes = len(np.unique(train_classes))
    print(num_classes)

    validation_classes = np.load(open('validation_classes.npy', 'rb'))

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'],
    )

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
    )

    generator_print_report(model, train_classes, train_generator, validation_classes, validation_generator)

    model.save_weights('first_try.h5')


def main():
    # save_bottleneck_features()
    # train_top_model()

    # fine_tune_convolution_block()

    small_conv_net_from_scratch()


if __name__ == '__main__':
    main()
