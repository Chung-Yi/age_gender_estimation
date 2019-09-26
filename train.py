import glob
import os
from argparse import ArgumentParser

import cv2
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

from cv_core.detector.face_inference import FaceLocationDetector
from keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, MaxPool2D
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils import np_utils

import tensorflow as tf

WEIGHT_INIT = "he_normal"
USE_BIAS = False
WEIGHT_DECAY = 0.0005
INPUT_SHAPE = (64, 64, 3)
AGE_CATEGORY = 100

BATCH_SIZE = 32
EPOCHS = 2000

parser = ArgumentParser()
parser.add_argument("train_dir", help="train dataset")
parser.add_argument("test_dir", help="test dataset")
parser.add_argument('--gpu', help="enable gpu", action='store_true')


def load_data(train_path):
    path_list = glob.glob(os.path.join(train_path, '**/*.jpg'), recursive=True)
    img_list = []
    age_list = []
    gender_list = []
    for path in path_list:
        filename = path.split('/')[-1]
        age, gender, _ = filename.split('_')
        img = cv2.imread(path)
        face_locs = detector.predict(img)
        if not 1 == len(face_locs):
            continue
        start_x, start_y, end_x, end_y = face_locs[0]
        img = img[start_y:end_y, start_x:end_x, :]
        img_list.append(cv2.resize(img, INPUT_SHAPE[:2]))
        age_list.append(int(age))
        gender_list.append(0 if gender == 'f' else 1)

    return np.asarray(img_list), age_list, gender_list


def freeze_session(session,
                   keep_var_names=None,
                   output_names=None,
                   clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(
            set(v.op.name for v in tf.global_variables()).difference(
                keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def init_detector():
    global detector

    detector = FaceLocationDetector()


def conv_block(inp, filters=32, bn=True, pool=True):
    m = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu')(inp)
    if bn:
        m = BatchNormalization()(m)
    if pool:
        m = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')(m)
    return m


def model_struct():
    inputs = Input(shape=INPUT_SHAPE)
    m = conv_block(inputs, filters=32, bn=False, pool=False)
    m = conv_block(m, filters=32 * 2)
    m = conv_block(m, filters=32 * 3)
    m = conv_block(m, filters=32 * 4)
    m = conv_block(m, filters=32 * 5)
    m = conv_block(m, filters=32 * 6)
    bottleneck = GlobalMaxPool2D()(m)

    # for age calculation
    a_dense = Dense(units=128, activation='relu')(bottleneck)
    age_output = Dense(
        units=AGE_CATEGORY, activation='sigmoid', name='age_output')(a_dense)

    # for gender prediction
    g_dense = Dense(units=128, activation='relu')(bottleneck)
    gender_output = Dense(
        units=2, activation='softmax', name='gender_output')(g_dense)

    model = Model(inputs=inputs, outputs=[gender_output, age_output])
    model.summary()

    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(
        optimizer='adam',
        loss=["categorical_crossentropy", "categorical_crossentropy"],
        metrics=['accuracy'])

    return model


def origin_model():
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
        activation="softmax")(flatten)
    predictions_a = Dense(
        units=AGE_CATEGORY,
        kernel_initializer=WEIGHT_INIT,
        use_bias=USE_BIAS,
        kernel_regularizer=l2(WEIGHT_DECAY),
        activation="softmax")(flatten)
    model = Model(inputs=inputs, outputs=[predictions_g, predictions_a])
    model.summary()

    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=sgd,
        loss=["categorical_crossentropy", "categorical_crossentropy"],
        metrics=['accuracy'])

    return model


def main():
    init_detector()
    args = parser.parse_args()
    train_path = args.train_dir
    test_path = args.test_dir

    if args.gpu:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.85
        K.tensorflow_backend.set_session(tf.Session(config=config))

    img, age, gender = load_data(train_path)
    x_train_data = img
    y_train_age = np_utils.to_categorical(age, AGE_CATEGORY)
    y_train_gender = np_utils.to_categorical(gender, 2)

    img, age, gender = load_data(test_path)
    x_test_data = img
    y_test_age = np_utils.to_categorical(age, AGE_CATEGORY)
    y_test_gender = np_utils.to_categorical(gender, 2)

    model = model_struct()
    

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=100,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True),
        ModelCheckpoint(
            "checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="auto")
    ]

    hist = model.fit(
        x_train_data, [y_train_gender, y_train_age],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=(x_test_data, [y_test_gender, y_test_age]))

    print('Train complete.')

    with K.get_session() as sess:
        # frozen graph
        additional_nodes = ['input_1', 'dense_1/Softmax', 'dense_2/Softmax']
        frozen_graph = freeze_session(sess, output_names=additional_nodes)

        # save model to pb file
        tf.train.write_graph(
            frozen_graph, "./", "age_gender_v1.pb", as_text=False)
        tf.train.write_graph(
            frozen_graph, "./", "age_gender_v1.pbtxt", as_text=True)


if __name__ == '__main__':
    main()
