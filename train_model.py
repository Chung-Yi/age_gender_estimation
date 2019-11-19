import numpy as np
import cv2
import albumentations as alb
import pandas as pd
import tensorflow as tf
import os
import glob
from argparse import ArgumentParser
from keras.utils import to_categorical
from PIL import Image

from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout, Flatten, Activation
from keras.optimizers import SGD, Adam
from keras.applications import ResNet50, InceptionResNetV2
from keras import regularizers
from keras.regularizers import l2, l1
from keras.models import Model
from keras import backend as K
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser
from albumentations import (HorizontalFlip, RandomBrightnessContrast)

IMG_PATH = 'faces/train_set'
WEIGHT_INIT = "he_normal"
USE_BIAS = False
WEIGHT_DECAY = 0.0005
INPUT_SHAPE = (64, 64, 3)
IMG_SIZE = 64
TRAIN_VAL_SPLIT = 0.8
MAXAGE = 70
EPOCHS = 2000
ID_GENDER_MAP = {0: 'f', 1: 'm'}
GENDER_ID_MAP = dict((g, i) for i, g in ID_GENDER_MAP.items())

parser = ArgumentParser()
parser.add_argument("model", help="choose a model")
parser.add_argument('--gpu', help="enable gpu", action='store_true')


def get_data_generator(df, indices, IMG_SIZE, for_training, batch_size=16):
    images, ages, genders = [], [], []

    while True:
        for i in indices:
            r = df.iloc[i]
            img_path, age, gender = r['img_path'], r['age'], r['gender_id']
            image = cv2.imread(img_path)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # image = Image.open(img_path)
            # image = image.resize((IMG_SIZE, IMG_SIZE))
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            aug_flip = HorizontalFlip(p=1)
            image_flip = aug_flip(image=image)['image']

            aug_brightness = alb.Compose([
                alb.RandomBrightnessContrast(p=1),
                alb.RandomGamma(p=1),
                alb.CLAHE(p=1),
            ],
                                         p=1)
            medium = alb.Compose([
                alb.CLAHE(p=1),
                alb.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=50,
                    val_shift_limit=50,
                    p=1),
            ],
                                 p=1)
            image_brightness = aug_brightness(image=image)['image']
            image_medium = medium(image=image)['image']

            image = np.array(image) / 255.0
            image_brightness = np.array(image_brightness) / 255.0
            image_flip = np.array(image_flip) / 255.0
            image_medium = np.array(image_medium) / 255.0
            images.append(image)
            images.append(image_brightness)
            images.append(image_flip)
            images.append(image_medium)
            ages.append(age / MAXAGE)
            ages.append(age / MAXAGE)
            ages.append(age / MAXAGE)
            ages.append(age / MAXAGE)
            genders.append(to_categorical(gender, 2))
            genders.append(to_categorical(gender, 2))
            genders.append(to_categorical(gender, 2))
            genders.append(to_categorical(gender, 2))

            if len(images) >= batch_size:
                yield np.array(images), [np.array(ages), np.array(genders)]
                images, ages, genders = [], [], []

        if not for_training:
            break


def parse_filepath(filepath):

    filename = filepath.split('/')[-1]
    age, gender, _ = filename.split('_')
    gender = 0 if gender == 'f' else 1

    return int(age), ID_GENDER_MAP[gender]


def model4():
    inputs = Input(shape=INPUT_SHAPE)
    model = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    model = Conv2D(32, (3, 3), padding='same', activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)

    model = Conv2D(64, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(64, (3, 3), padding='same', activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)

    model = Conv2D(128, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(128, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(128, (3, 3), padding='same', activation='relu')(model)
    model = BatchNormalization(axis=-1)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)

    model = Conv2D(128, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(128, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(128, (3, 3), padding='same', activation='relu')(model)
    model = BatchNormalization(axis=-1)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)

    model = Flatten()(model)

    model = Dense(256, activation='relu')(model)
    model = Dense(256, activation='relu')(model)

    model = Dense(128, activation='relu')(model)
    model = Dense(64, activation='relu')(model)

    predictions_g = Dense(units=2, activation='softmax', name='gender')(model)
    predictions_a = Dense(units=1, activation='sigmoid', name='age')(model)

    model = Model(inputs=inputs, outputs=[predictions_a, predictions_g])
    model.summary()

    sgd = SGD(lr=0.01, momentum=0.9)
    losses = {
        "gender": "categorical_crossentropy",
        "age": "mse",
    }
    model.compile(
        optimizer='adam',
        loss=losses,
        metrics={
            'age': 'mae',
            'gender': 'accuracy'
        })

    return model


def model3():
    inputs = Input(shape=INPUT_SHAPE)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    pool = AveragePooling2D(
        pool_size=(4, 4), strides=(1, 1), padding="same")(x)
    flatten = Flatten()(pool)
    predictions_g = Dense(
        units=2,
        kernel_initializer=WEIGHT_INIT,
        use_bias=USE_BIAS,
        kernel_regularizer=l2(WEIGHT_DECAY),
        activation="softmax",
        name='gender')(flatten)
    predictions_a = Dense(
        units=1,
        kernel_initializer=WEIGHT_INIT,
        use_bias=USE_BIAS,
        kernel_regularizer=l2(WEIGHT_DECAY),
        activation="sigmoid",
        name='age')(flatten)
    model = Model(inputs=inputs, outputs=[predictions_a, predictions_g])
    model.summary()

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    losses = {"gender": "categorical_crossentropy", "age": "mse"}
    model.compile(
        optimizer='adam',
        loss=losses,
        metrics={
            'age': 'mae',
            'gender': 'accuracy'
        })

    return model


def model2():
    inputs = Input(shape=INPUT_SHAPE)
    g = Conv2D(32, (3, 3), padding="same")(inputs)
    g = Activation("relu")(g)
    g = BatchNormalization(axis=-1)(g)
    g = MaxPooling2D(pool_size=(3, 3))(g)
    g = Dropout(0.25)(g)

    g = Conv2D(64, (3, 3), padding="same")(g)
    g = Activation("relu")(g)
    g = BatchNormalization(axis=-1)(g)
    g = MaxPooling2D(pool_size=(2, 2))(g)
    g = Dropout(0.25)(g)

    g = Conv2D(32, (3, 3), padding="same")(g)
    g = Activation("relu")(g)
    g = BatchNormalization(axis=-1)(g)
    g = MaxPooling2D(pool_size=(2, 2))(g)
    g = Dropout(0.25)(g)

    g = Flatten()(g)
    g = Dense(256)(g)
    g = Activation("relu")(g)
    g = BatchNormalization()(g)
    g = Dropout(0.5)(g)
    predictions_g = Dense(2, activation='softmax', name='gender')(g)

    a = Conv2D(32, (3, 3), padding="same")(inputs)
    a = Activation("relu")(a)
    a = BatchNormalization(axis=-1)(a)
    a = MaxPooling2D(pool_size=(3, 3))(a)
    a = Dropout(0.25)(a)

    a = Conv2D(64, (3, 3), padding="same")(a)
    a = Activation("relu")(a)
    a = BatchNormalization(axis=-1)(a)
    a = Conv2D(64, (3, 3), padding="same")(a)
    a = Activation("relu")(a)
    a = BatchNormalization(axis=-1)(a)
    a = MaxPooling2D(pool_size=(2, 2))(a)
    a = Dropout(0.25)(a)

    a = Conv2D(128, (3, 3), padding="same")(a)
    a = Activation("relu")(a)
    a = BatchNormalization(axis=-1)(a)
    a = Conv2D(128, (3, 3), padding="same")(a)
    a = Activation("relu")(a)
    a = BatchNormalization(axis=-1)(a)
    a = MaxPooling2D(pool_size=(2, 2))(a)
    a = Dropout(0.25)(a)

    a = Flatten()(a)
    a = Dense(256)(a)
    a = Activation("relu")(a)
    a = BatchNormalization()(a)
    a = Dropout(0.5)(a)
    predictions_a = Dense(1, activation='sigmoid', name='age')(a)

    model = Model(inputs=inputs, outputs=[predictions_a, predictions_g])
    model.summary()

    opt = Adam(lr=0.001)
    losses = {
        "age": "mse",
        "gender": "categorical_crossentropy",
    }

    model.compile(
        optimizer=opt,
        loss=losses,
        metrics={
            'age': 'mae',
            'gender': 'accuracy'
        })
    return model


def model1():
    inputs = Input(shape=INPUT_SHAPE)
    model = Conv2D(
        32,
        padding='same',
        kernel_size=(3, 3),
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01))(inputs)
    # model = Dropout(0.2)(model)
    model = Conv2D(
        32, padding='same', kernel_size=(3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Conv2D(
        64, padding='same', kernel_size=(3, 3), activation='relu')(model)
    # model = Dropout(0.2)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Conv2D(
        64, padding='same', kernel_size=(3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Conv2D(
        128, padding='same', kernel_size=(3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Conv2D(
        128, padding='same', kernel_size=(3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)

    model = Flatten()(model)

    model = Dense(256, activation='relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(64, activation='relu')(model)

    predictions_g = Dense(
        units=2,
        kernel_initializer=WEIGHT_INIT,
        use_bias=USE_BIAS,
        kernel_regularizer=l2(WEIGHT_DECAY),
        activation="softmax",
        name='gender')(model)
    predictions_a = Dense(
        units=1,
        kernel_initializer=WEIGHT_INIT,
        use_bias=USE_BIAS,
        kernel_regularizer=l2(WEIGHT_DECAY),
        activation="sigmoid",
        name='age')(model)
    model = Model(inputs=inputs, outputs=[predictions_a, predictions_g])
    model.summary()

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=sgd,
        loss=["mse", "categorical_crossentropy"],
        metrics={
            'age': 'mae',
            'gender': 'accuracy'
        })

    return model


def main():

    args = parser.parse_args()
    if args.gpu:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.85
        K.tensorflow_backend.set_session(tf.Session(config=config))

    img_list = glob.glob(os.path.join(IMG_PATH, '*.jpg'))
    attributes = list(map(parse_filepath, img_list))
    df = pd.DataFrame(attributes)
    df['img_path'] = img_list
    df.columns = ['age', 'gender', 'img_path']
    df = df.dropna()
    # print(df.head())

    p = np.random.permutation(len(img_list))

    train_up_to = int(len(img_list) * TRAIN_VAL_SPLIT)

    train_idx = p[:train_up_to]

    valid_idx = p[train_up_to:]

    df['gender_id'] = df['gender'].map(lambda gender: GENDER_ID_MAP[gender])

    batch_size = 64
    valid_batch_size = 64
    train_gen = get_data_generator(
        df, train_idx, IMG_SIZE, for_training=True, batch_size=batch_size)
    valid_gen = get_data_generator(
        df,
        valid_idx,
        IMG_SIZE,
        for_training=True,
        batch_size=valid_batch_size)

    # # training
    # input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # _ = conv_block(input_layer, filters=32, bn=False, pool=False)
    # _ = conv_block(_, filters=32 * 2)
    # _ = conv_block(_, filters=32 * 4)
    # _ = conv_block(_, filters=32 * 6)
    # _ = conv_block(_, filters=32 * 8)
    # # _ = conv_block(_, filters=32 * 6)
    # gm = GlobalMaxPool2D()(_)

    # # for age calculation
    # _ = Dense(units=128, activation='relu')(gm)
    # age_output = Dense(units=1, activation='sigmoid', name='age_output')(_)

    # # for gender prediction
    # _ = Dense(units=128, activation='relu')(gm)
    # gender_output = Dense(
    #     units=len(GENDER_ID_MAP), activation='softmax',
    #     name='gender_output')(_)

    # model = Model(inputs=input_layer, outputs=[age_output, gender_output])
    # model.compile(
    #     optimizer='adam',
    #     loss={
    #         'age_output': 'mse',
    #         'gender_output': 'categorical_crossentropy'
    #     },
    #     loss_weights={
    #         'age_output': 2.,
    #         'gender_output': 1.
    #     },
    #     metrics={
    #         'age_output': 'mae',
    #         'gender_output': 'accuracy'
    #     })

    # model.summary()

    if args.model == 'model1':
        model = model1()
    elif args.model == 'model2':
        model = model2()
    elif args.model == 'model3':
        model = model3()
    elif args.model == 'model4':
        model = model4()
    else:
        model = multi_model()

    batch_size = 64
    valid_batch_size = 64

    # callbacks = [ModelCheckpoint("./model_checkpoint", monitor='val_loss')]
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=100,
            verbose=0,
            mode='min',
            baseline=None,
            restore_best_weights=True),
        ModelCheckpoint(
            # "checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
            'models/{epoch:02d}-{val_loss:.2f}.h5',
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min")
    ]

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=len(train_idx) // batch_size,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_gen,
        validation_steps=len(valid_idx) // valid_batch_size)


if __name__ == '__main__':
    main()