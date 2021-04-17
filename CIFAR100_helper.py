import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Activation, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
import tensorboard
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from common import *


def create_superclass(x):
    fine_id_coarse_id = {0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11, 20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15, 30: 0,
                         31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16,
                         37: 9, 38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10, 50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17, 60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16,
                         66: 12, 67: 1, 68: 9, 69: 19, 70: 2, 71: 10, 72: 0, 73: 1, 74: 16,
                         75: 12, 76: 9, 77: 13, 78: 15, 79: 13, 80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19, 90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13}
    return fine_id_coarse_id[x]


v_create_superclass = np.vectorize(create_superclass)


def create_ultraclass(x):
    coarse_id_ultra_id = {0: 0, 1: 4, 2: 1, 3: 2, 4: 1, 5: 2, 6: 2, 7: 3, 8: 4, 9: 6, 10: 6, 11: 4, 12: 0, 13: 3, 14: 0, 15: 4, 16: 0, 17: 6, 18: 5, 19: 5}
    return coarse_id_ultra_id[x]


v_create_ultraclass = np.vectorize(create_ultraclass)


def get_cifar100():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode='fine')
    X, Y_sub = np.concatenate((x_train, x_test), axis=0) / 255, np.concatenate((y_train, y_test), axis=0)
    Y_super = v_create_superclass(Y_sub)
    Y_ultra = v_create_ultraclass(Y_super)

    return X, to_categorical(Y_ultra), to_categorical(Y_super), to_categorical(Y_sub)


def get_cifar100_with_split():
    (x_train, y_sub_train), (x_test, y_sub_test) = keras.datasets.cifar100.load_data(label_mode='fine')
    x_train, x_test = x_train / 255, x_test / 255
    y_super_train, y_super_test = v_create_superclass(y_sub_train), v_create_superclass(y_sub_test)
    y_ultra_train, y_ultra_test = v_create_ultraclass(y_super_train), v_create_ultraclass(y_super_test)

    return (x_train, to_categorical(y_ultra_train), to_categorical(y_super_train), to_categorical(y_sub_train)), (x_test, to_categorical(y_ultra_test), to_categorical(y_super_test), to_categorical(y_sub_test))


def get_labels(idx, level='super'):
    labels = {
        'ultra': ['mammal', 'plant', 'household', 'invertebrates', 'other animal', 'vehicles', 'nature'],

        'super': ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit & vegetables',
                  'household electrical device', 'household furniture', 'insects', 'large carnivores',
                  'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
                  'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2'
                  ],
        'sub': [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
    }
    return labels[level][idx]


v_get_labels = np.vectorize(get_labels)


def plot_sample_images(inputs, v_label_func, top_n=14, n_col=7):
    (images, ultra_labels_idx, super_labels_idx, sub_labels_idx) = inputs
    assert len(images) == len(ultra_labels_idx) == len(super_labels_idx) == len(sub_labels_idx)
    ultra_labels, super_labels, sub_labels = v_label_func(np.argmax(ultra_labels_idx, axis=1), 'ultra'), v_label_func(np.argmax(super_labels_idx, axis=1), 'super'), v_label_func(np.argmax(sub_labels_idx, axis=1), 'sub')
    top_n = top_n if top_n < len(images) else len(images)
    n_row = int(top_n / n_col)
    images = images[:top_n]
    figsize = (4 * n_col, 3.5 * n_row)
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    for idx, image in enumerate(images):
        ax = axes[idx // n_col, idx % n_col]
        ax.imshow(image)
        ax.set_title(f"{ultra_labels[idx]} - {super_labels[idx]} - {sub_labels[idx]}")
    plt.tight_layout()
    plt.show()


def train_and_predict(model, epochs, train, test, batch_size=256):
    (x_train, y_ultra_train, y_super_train, y_sub_train), (x_test, y_ultra_test, y_super_test, y_sub_test) = train, test
    model.fit(
        x_train,
        y={"ultra_output": y_ultra_train, "super_output": y_super_train, "sub_output": y_sub_train},
        validation_data=(x_test, {"ultra_output": y_ultra_test, "super_output": y_super_test, "sub_output": y_sub_test}),
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        use_multiprocessing=True,
    )

    yhat_ultra_test, yhat_super_test, yhat_sub_test = model.predict(x_test, use_multiprocessing=True)
    y_ultra, yhat_ultra = v_get_labels(np.argmax(y_ultra_test, axis=1), 'ultra'), v_get_labels(np.argmax(yhat_ultra_test, axis=1), 'ultra')
    y_super, yhat_super = v_get_labels(np.argmax(y_super_test, axis=1), 'super'), v_get_labels(np.argmax(yhat_super_test, axis=1), 'super')
    y_sub, yhat_sub = v_get_labels(np.argmax(y_sub_test, axis=1), 'sub'), v_get_labels(np.argmax(yhat_sub_test, axis=1), 'sub')

    # Calculate Metrics
    n_rows = len(x_test)
    error_rate_ultra = sum(y_ultra != yhat_ultra) / n_rows
    error_rate_super = sum(y_super != yhat_super) / n_rows
    error_rate_sub = sum(y_sub != yhat_sub) / n_rows

    mismatch_rate = 0

    hierarchy = gen_hierarchy(y_ultra_train, y_sub_train)
    n_mismatch = calculate_hiearchy_mismatch(np.argmax(yhat_ultra_test, axis=1), np.argmax(yhat_sub_test, axis=1), hierarchy)
    mismatch_rate += n_mismatch / n_rows

    hierarchy = gen_hierarchy(y_ultra_train, y_super_train)
    n_mismatch = calculate_hiearchy_mismatch(np.argmax(yhat_ultra_test, axis=1), np.argmax(yhat_super_test, axis=1), hierarchy)
    mismatch_rate += n_mismatch / n_rows

    hierarchy = gen_hierarchy(y_super_train, y_sub_train)
    n_mismatch = calculate_hiearchy_mismatch(np.argmax(yhat_super_test, axis=1), np.argmax(yhat_sub_test, axis=1), hierarchy)
    mismatch_rate += n_mismatch / n_rows

    mismatch_rate /= 3

    print(f"Ultra Class Error Rate: {error_rate_ultra}\nSuper Class Error Rate: {error_rate_super}\nSub Class Error Rate: {error_rate_sub}\nAverage Mismatch Rate: {mismatch_rate}")
    return error_rate_ultra, error_rate_super, error_rate_sub, mismatch_rate


def train_and_predict_bottom_up(model, epochs, train, test, batch_size=256):
    (x_train, y_ultra_train, y_super_train, y_sub_train), (x_test, y_ultra_test, y_super_test, y_sub_test) = train, test
    model.fit(
        x_train,
        y=y_sub_train,
        validation_data=(x_test, y_sub_test),
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        use_multiprocessing=True,
    )

    yhat_sub_test = np.argmax(model.predict(x_test, use_multiprocessing=True), axis=1)
    yhat_super_test = v_create_superclass(yhat_sub_test)
    yhat_ultra_test = v_create_ultraclass(yhat_super_test)
    y_ultra, yhat_ultra = v_get_labels(np.argmax(y_ultra_test, axis=1), 'ultra'), v_get_labels(yhat_ultra_test, 'ultra')
    y_super, yhat_super = v_get_labels(np.argmax(y_super_test, axis=1), 'super'), v_get_labels(yhat_super_test, 'super')
    y_sub, yhat_sub = v_get_labels(np.argmax(y_sub_test, axis=1), 'sub'), v_get_labels(yhat_sub_test, 'sub')

    # Calculate Metrics
    n_rows = len(x_test)
    error_rate_ultra = sum(y_ultra != yhat_ultra) / n_rows
    error_rate_super = sum(y_super != yhat_super) / n_rows
    error_rate_sub = sum(y_sub != yhat_sub) / n_rows

    mismatch_rate = 0

    hierarchy = gen_hierarchy(y_ultra_train, y_sub_train)
    n_mismatch = calculate_hiearchy_mismatch(yhat_ultra_test, yhat_sub_test, hierarchy)
    mismatch_rate += n_mismatch / n_rows

    hierarchy = gen_hierarchy(y_ultra_train, y_super_train)
    n_mismatch = calculate_hiearchy_mismatch(yhat_ultra_test, yhat_super_test, hierarchy)
    mismatch_rate += n_mismatch / n_rows

    hierarchy = gen_hierarchy(y_super_train, y_sub_train)
    n_mismatch = calculate_hiearchy_mismatch(yhat_super_test, yhat_sub_test, hierarchy)
    mismatch_rate += n_mismatch / n_rows

    mismatch_rate /= 3

    print(f"Ultra Class Error Rate: {error_rate_ultra}\nSuper Class Error Rate: {error_rate_super}\nSub Class Error Rate: {error_rate_sub}\nAverage Mismatch Rate: {mismatch_rate}")
    return error_rate_ultra, error_rate_super, error_rate_sub, mismatch_rate